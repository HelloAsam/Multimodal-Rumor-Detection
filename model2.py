# ============================================
# 模块 1. 加载依赖
# ============================================
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor, ViTModel

# ============================================
# 模块 2. 设置参数 & 路径
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
BATCH_SIZE = 8

DATA_DIR = "./data"
IMAGE_ROOT = "./images"

# 模型路径（你之前已下载）
BERT_PATH = "./pretrained/chinese-bert-wwm-ext"
VIT_PATH = "./pretrained/vit-base-patch16-224"

# ============================================
# 模块 3. 加载预训练模型
# ============================================
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert_model = AutoModel.from_pretrained(BERT_PATH).to(DEVICE)

vit_processor = ViTImageProcessor.from_pretrained(VIT_PATH)
vit_model = ViTModel.from_pretrained(VIT_PATH).to(DEVICE)

# ============================================
# 模块 4. 生成统一 DataFrame
# ============================================
def load_meta(meta_path, social_path, split_name):
    """
    从 meta.xlsx 和 social_feature.xlsx 构造统一 DataFrame
    """
    # 读取 meta（含 weibo_id, text, image_files, label）
    df = pd.read_excel(meta_path)

    # 读取社交特征
    social_df = pd.read_excel(social_path)

    # weibo_id 转字符串，避免科学计数法
    df["weibo_id"] = df["weibo_id"].astype(str)
    social_df["weibo_id"] = social_df["weibo_id"].astype(str)

    # 合并 (inner join)
    df = df.merge(social_df, on="weibo_id", how="inner")

    # 提取社交特征列
    feature_cols = [c for c in df.columns if c not in ["weibo_id", "text", "image_files", "image_count", "label"]]
    df["social_features"] = df[feature_cols].values.tolist()

    # 保留关键列
    df = df[["weibo_id", "text", "image_files", "social_features", "label"]]

    print(f"{split_name} 样本数: {len(df)}")
    return df

social_path = os.path.join(DATA_DIR, "social_feature.xlsx")
train_df = load_meta(os.path.join(DATA_DIR, "train_meta.xlsx"), social_path, "训练集")
val_df   = load_meta(os.path.join(DATA_DIR, "val_meta.xlsx"), social_path, "验证集")
test_df  = load_meta(os.path.join(DATA_DIR, "test_meta.xlsx"), social_path, "测试集")

# ============================================
# 模块 5. 定义 Dataset
# ============================================
class WeiboDataset(Dataset):
    def __init__(self, df, text_tokenizer, image_processor, image_root=IMAGE_ROOT, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.image_root = image_root
        self.max_len = max_len

        # 调试：打印空文本数量
        empty_count = self.df["text"].isna().sum()
        print(f"📊 当前数据集中空文本数量: {empty_count}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ========= 文本处理 =========
        text = row["text"]
        if not isinstance(text, str):
            text = ""   # 避免 NaN 或数字报错

        text_inputs = self.text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # ========= 图像处理（多图 → mean pooling）=========
        img_files = str(row["image_files"]).split(",")
        img_list = []
        for fname in img_files:
            path = os.path.join(self.image_root, fname.strip())
            if not os.path.exists(path):
                continue
            try:
                image = Image.open(path).convert("RGB")
                inputs = self.image_processor(images=image, return_tensors="pt")
                img_list.append(inputs["pixel_values"].squeeze(0))
            except Exception as e:
                print(f"❌ 图像读取失败: {path}, 错误: {e}")
                continue

        if len(img_list) == 0:
            img_tensor = torch.zeros(3, 224, 224)  # 占位
        else:
            img_tensor = torch.stack(img_list).mean(dim=0)

        # ========= 社交特征 =========
        social_feat = torch.tensor(row["social_features"], dtype=torch.float)

        label = int(row["label"])

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "pixel_values": img_tensor,
            "social_features": social_feat,
            "label": label
        }


# ============================================
# 模块 6. 定义多模态模型（支持部分解冻）
# ============================================
class MultiModalModel(nn.Module):
    def __init__(self, bert, vit, social_dim=16, hidden_dim=256, num_classes=2,
                 freeze_backbone=True, unfreeze_last_n=0):
        super().__init__()
        self.bert = bert
        self.vit = vit

        # ===== 先冻结所有参数 =====
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.vit.parameters():
                param.requires_grad = False

        # ===== 如果指定 unfreeze_last_n > 0，就解冻 BERT / ViT 的后几层 =====
        if unfreeze_last_n > 0:
            # BERT: encoder.layer[-n:]
            for layer in self.bert.encoder.layer[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # BERT pooler 也解冻（可选）
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

            # ViT: encoder.layer[-n:]
            for layer in self.vit.encoder.layer[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # ViT pooler 也解冻（可选）
            for param in self.vit.pooler.parameters():
                param.requires_grad = True

        # ===== 投影层 =====
        self.text_proj = nn.Linear(768, hidden_dim)
        self.img_proj = nn.Linear(768, hidden_dim)
        self.social_proj = nn.Linear(social_dim, hidden_dim)

        # ===== 三层 MLP 分类头 =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # ===== 打印参数信息 =====
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        def format_params(n):
            if n >= 1e6:
                return f"{n/1e6:.2f}M"
            elif n >= 1e3:
                return f"{n/1e3:.2f}K"
            return str(n)

        print("📊 模型参数量统计:")
        print(f"  总参数: {total_params} ({format_params(total_params)})")
        print(f"  可训练参数: {trainable_params} ({format_params(trainable_params)})")
        print(f"  冻结参数: {total_params - trainable_params} ({format_params(total_params - trainable_params)})")

    def forward(self, input_ids, attention_mask, pixel_values, social_features):
        # 文本特征
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_proj(text_out.pooler_output)  # [B, hidden_dim]

        # 图像特征
        img_out = self.vit(pixel_values=pixel_values)
        img_feat = self.img_proj(img_out.pooler_output)  # [B, hidden_dim]

        # 社交特征
        social_feat = self.social_proj(social_features)

        # 融合
        fused = torch.cat([text_feat, img_feat, social_feat], dim=1)

        return self.classifier(fused)



# ============================================
# 模块 7. 数据加载器
# ============================================
train_dataset = WeiboDataset(train_df, bert_tokenizer, vit_processor)
val_dataset   = WeiboDataset(val_df, bert_tokenizer, vit_processor)
test_dataset  = WeiboDataset(test_df, bert_tokenizer, vit_processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ============================================
# 模块 8. 模型测试前向
# ============================================
# 解冻 BERT 和 ViT 的后两层
model = MultiModalModel(
    bert_model,
    vit_model,
    freeze_backbone=True,   # 先冻结
    unfreeze_last_n=2       # 再解冻后两层
).to(DEVICE)

# 取一个 batch 测试 forward
batch = next(iter(train_loader))
out = model(
    batch["input_ids"].to(DEVICE),
    batch["attention_mask"].to(DEVICE),
    batch["pixel_values"].to(DEVICE),
    batch["social_features"].to(DEVICE)
)

print("输出维度:", out.shape)  # [B, 2]



# ============================================
# 模块 9. 训练 & 验证函数 (增强版: 输出分类指标)
# ============================================
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

def get_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=["nonrumor(0)", "rumor(1)"], digits=4, output_dict=True)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision_0": report["nonrumor(0)"]["precision"],
        "recall_0": report["nonrumor(0)"]["recall"],
        "f1_0": report["nonrumor(0)"]["f1-score"],
        "precision_1": report["rumor(1)"]["precision"],
        "recall_1": report["rumor(1)"]["recall"],
        "f1_1": report["rumor(1)"]["f1-score"],
    }

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []

    for batch in dataloader:
        optimizer.zero_grad()

        outputs = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["pixel_values"].to(DEVICE),
            batch["social_features"].to(DEVICE)
        )

        labels = batch["label"].to(DEVICE)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds_all.extend(outputs.argmax(1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    metrics = get_metrics(labels_all, preds_all)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["pixel_values"].to(DEVICE),
                batch["social_features"].to(DEVICE)
            )
            labels = batch["label"].to(DEVICE)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds_all.extend(outputs.argmax(1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    metrics = get_metrics(labels_all, preds_all)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


# ============================================
# 模块 10. 训练主循环 (早停 + 日志增强版)
# ============================================
import csv

EPOCHS = 30
LR = 2e-5
PATIENCE = 5

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)

best_val_f1 = 0.0
patience_counter = 0

# 确保文件夹存在
os.makedirs("model", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

# 日志文件路径
log_path = os.path.join("tmp", "training_log.csv")

# 写入表头
with open(log_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    headers = [
        "epoch",
        # train
        "train_loss", "train_acc", "train_macro_f1",
        "train_precision_0", "train_recall_0", "train_f1_0",
        "train_precision_1", "train_recall_1", "train_f1_1",
        # val
        "val_loss", "val_acc", "val_macro_f1",
        "val_precision_0", "val_recall_0", "val_f1_0",
        "val_precision_1", "val_recall_1", "val_f1_1"
    ]
    writer.writerow(headers)

for epoch in range(EPOCHS):
    print(f"\n🔹 Epoch {epoch+1}/{EPOCHS}")

    train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
    val_metrics = evaluate(model, val_loader, criterion)

    print(f"训练集: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['acc']:.4f}, F1={train_metrics['macro_f1']:.4f}")
    print(f"验证集: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.4f}, F1={val_metrics['macro_f1']:.4f}")

    # 保存日志
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [epoch+1] + [
            train_metrics["loss"], train_metrics["acc"], train_metrics["macro_f1"],
            train_metrics["precision_0"], train_metrics["recall_0"], train_metrics["f1_0"],
            train_metrics["precision_1"], train_metrics["recall_1"], train_metrics["f1_1"],
            val_metrics["loss"], val_metrics["acc"], val_metrics["macro_f1"],
            val_metrics["precision_0"], val_metrics["recall_0"], val_metrics["f1_0"],
            val_metrics["precision_1"], val_metrics["recall_1"], val_metrics["f1_1"]
        ]
        writer.writerow(row)

    # 早停机制
    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        torch.save(model.state_dict(), os.path.join("model", "best_multimodal_model.pth"))
        print("✅ 最佳模型已保存到 ./model/best_multimodal_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"⚠️ 验证 F1 没提升, patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("⏹️ 触发早停机制，训练提前结束")
            break

# ============================================
# 模块 11. 测试集评估 (结果写入 training_log.csv + 保存混淆矩阵)
# ============================================
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print("\n=== 在测试集上评估 ===")
model.load_state_dict(torch.load(os.path.join("model", "best_multimodal_model.pth")))
test_metrics = evaluate(model, test_loader, criterion)

print(f"\n📊 测试集整体表现: Loss={test_metrics['loss']:.4f}, "
      f"Acc={test_metrics['acc']:.4f}, Macro-F1={test_metrics['macro_f1']:.4f}")

# 收集预测与标签
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["pixel_values"].to(DEVICE),
            batch["social_features"].to(DEVICE)
        )
        preds = outputs.argmax(1).cpu().numpy()
        labels = batch["label"].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

# 分类报告
report = classification_report(
    all_labels, all_preds,
    target_names=["nonrumor(0)", "rumor(1)"],
    digits=4
)
print("\n📑 分类报告:")
print(report)

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
print("\n🧩 混淆矩阵:")
print(cm)

# 保存混淆矩阵
os.makedirs("tmp", exist_ok=True)
np.save(os.path.join("tmp", "confusion_matrix.npy"), cm)
print("✅ 混淆矩阵已保存到 tmp/confusion_matrix.npy")

# === 把测试集结果写入 training_log.csv ===
with open(log_path, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    row = ["TEST"] + [
        test_metrics["loss"], test_metrics["acc"], test_metrics["macro_f1"],
        test_metrics["precision_0"], test_metrics["recall_0"], test_metrics["f1_0"],
        test_metrics["precision_1"], test_metrics["recall_1"], test_metrics["f1_1"],
        "-", "-", "-", "-", "-", "-", "-", "-", "-"
    ]
    writer.writerow(row)

print(f"\n✅ 测试结果已追加到 {log_path}")
