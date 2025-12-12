# ============================================
# 模块 1. 加载依赖
# ============================================
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    ViTImageProcessor, ViTModel,
    CLIPProcessor, CLIPModel
)

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import csv

# ============================================
# 模块 2. 设置参数 & 路径
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
BATCH_SIZE = 8

DATA_DIR = "./data"
IMAGE_ROOT = "./images"

# 模型路径（需提前下载）
BERT_PATH = "./pretrained/chinese-bert-wwm-ext"
ROBERTA_PATH = "./pretrained/chinese-roberta-wwm-ext"
VIT_PATH = "./pretrained/vit-base-patch16-224"
CLIP_PATH = "./pretrained/clip-vit-base-patch32"

# ============================================
# 模块 3. 加载预训练模型
# ============================================
# 文本模型
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert_model = AutoModel.from_pretrained(BERT_PATH).to(DEVICE)

roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
roberta_model = AutoModel.from_pretrained(ROBERTA_PATH).to(DEVICE)

# 图像模型
vit_processor = ViTImageProcessor.from_pretrained(VIT_PATH)
vit_model = ViTModel.from_pretrained(VIT_PATH).to(DEVICE)

clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)
clip_model = CLIPModel.from_pretrained(CLIP_PATH).to(DEVICE)
clip_visual = clip_model.vision_model  # 只保留视觉编码器

# ============================================
# 模块 4. 加载 meta 与社交特征
# ============================================
def load_meta(meta_path, social_path, split_name):
    df = pd.read_excel(meta_path)
    social_df = pd.read_excel(social_path)

    df["weibo_id"] = df["weibo_id"].astype(str)
    social_df["weibo_id"] = social_df["weibo_id"].astype(str)

    df = df.merge(social_df, on="weibo_id", how="inner")
    feature_cols = [c for c in df.columns if c not in ["weibo_id", "text", "image_files", "image_count", "label"]]
    df["social_features"] = df[feature_cols].values.tolist()
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
    def __init__(self, df, bert_tokenizer, roberta_tokenizer, vit_processor, clip_processor, image_root=IMAGE_ROOT, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.bert_tokenizer = bert_tokenizer
        self.roberta_tokenizer = roberta_tokenizer
        self.vit_processor = vit_processor
        self.clip_processor = clip_processor
        self.image_root = image_root
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"] if isinstance(row["text"], str) else ""

        # BERT 输入
        bert_inputs = self.bert_tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        # RoBERTa 输入
        roberta_inputs = self.roberta_tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        # 图像输入
        img_files = str(row["image_files"]).split(",")
        vit_imgs, clip_imgs = [], []
        for fname in img_files:
            path = os.path.join(self.image_root, fname.strip())
            if not os.path.exists(path):
                continue
            try:
                image = Image.open(path).convert("RGB")
                vit_inputs = self.vit_processor(images=image, return_tensors="pt")
                clip_inputs = self.clip_processor(images=image, return_tensors="pt")

                vit_imgs.append(vit_inputs["pixel_values"].squeeze(0))
                clip_imgs.append(clip_inputs["pixel_values"].squeeze(0))
            except:
                continue

        if len(vit_imgs) == 0:
            vit_tensor = torch.zeros(3, 224, 224)
            clip_tensor = torch.zeros(3, 224, 224)
        else:
            vit_tensor = torch.stack(vit_imgs).mean(dim=0)
            clip_tensor = torch.stack(clip_imgs).mean(dim=0)

        social_feat = torch.tensor(row["social_features"], dtype=torch.float)
        label = int(row["label"])

        return {
            "bert_input_ids": bert_inputs["input_ids"].squeeze(0),
            "bert_attention_mask": bert_inputs["attention_mask"].squeeze(0),
            "roberta_input_ids": roberta_inputs["input_ids"].squeeze(0),
            "roberta_attention_mask": roberta_inputs["attention_mask"].squeeze(0),
            "vit_pixel_values": vit_tensor,
            "clip_pixel_values": clip_tensor,
            "social_features": social_feat,
            "label": label
        }

# ============================================
# 模块 6. 定义四模态模型 (Text/Image 各自 concat + 堆叠双向MHCA + 与社交特征拼接)
# 仍然仅解冻每个预训练模型最后 unfreeze_layers 层（默认2层）
# ============================================
class CrossModalBlock(nn.Module):
    """
    双向跨模态注意力块：
    - Text <- Image (Text as Q; Image as K/V)
    - Image <- Text (Image as Q; Text as K/V)
    每条路径都有：MHA + 残差 + LN + FFN
    预期输入:
      text:  [B, Lt, C]  (此处 Lt=1)
      image: [B, Li, C]  (此处 Li=1)
    """
    def __init__(self, embed_dim=512, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.ln_t_q = nn.LayerNorm(embed_dim)
        self.ln_i_kv = nn.LayerNorm(embed_dim)
        self.mha_t_to_i = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ln_i_q = nn.LayerNorm(embed_dim)
        self.ln_t_kv = nn.LayerNorm(embed_dim)
        self.mha_i_to_t = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        ff_dim = embed_dim * ff_mult
        self.ffn_t = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_i = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, text, image):
        # Text ← Image
        t_in = text
        t_q = self.ln_t_q(text)
        i_kv = self.ln_i_kv(image)
        t_ctx, _ = self.mha_t_to_i(t_q, i_kv, i_kv)  # [B, Lt, C]
        text = t_in + t_ctx
        text = text + self.ffn_t(text)

        # Image ← Text
        i_in = image
        i_q = self.ln_i_q(image)
        t_kv = self.ln_t_kv(text)
        i_ctx, _ = self.mha_i_to_t(i_q, t_kv, t_kv)  # [B, Li, C]
        image = i_in + i_ctx
        image = image + self.ffn_i(image)

        return text, image


class MultiModalModel(nn.Module):
    def __init__(
        self, bert, roberta, vit, clip_visual,
        social_dim=16, hidden_dim=256, num_classes=2,
        unfreeze_layers=2,              # 仍然只解冻最后2层
        xattn_layers=2,                 # MHCA 堆叠层数，可调 2~3
        xattn_heads=8,                  # 多头数
        xattn_ff_mult=4,                # FFN 放大倍数
        xattn_dropout=0.1
    ):
        super().__init__()
        self.bert = bert
        self.roberta = roberta
        self.vit = vit
        self.clip_visual = clip_visual

        # ===== 1) 先冻后解：只解冻每个 backbone 的最后 N 层 =====
        for model in [self.bert, self.roberta, self.vit, self.clip_visual]:
            for p in model.parameters():
                p.requires_grad = False

        if hasattr(self.bert, "encoder"):
            for layer in self.bert.encoder.layer[-unfreeze_layers:]:
                for p in layer.parameters(): p.requires_grad = True
        if hasattr(self.roberta, "encoder"):
            for layer in self.roberta.encoder.layer[-unfreeze_layers:]:
                for p in layer.parameters(): p.requires_grad = True
        if hasattr(self.vit, "encoder"):
            for layer in self.vit.encoder.layer[-unfreeze_layers:]:
                for p in layer.parameters(): p.requires_grad = True
        if hasattr(self.clip_visual, "encoder"):
            for layer in self.clip_visual.encoder.layers[-unfreeze_layers:]:
                for p in layer.parameters(): p.requires_grad = True

        # ===== 2) 投影到统一维度，然后各自 concat 成 512 维 =====
        self.bert_proj   = nn.Linear(768, hidden_dim)  # -> 256
        self.roberta_proj= nn.Linear(768, hidden_dim)  # -> 256
        self.vit_proj    = nn.Linear(768, hidden_dim)  # -> 256
        self.clip_proj   = nn.Linear(768, hidden_dim)  # -> 256
        self.social_proj = nn.Linear(social_dim, hidden_dim)  # -> 256

        # ===== 3) 堆叠 N 层双向 MHCA =====
        self.xblocks = nn.ModuleList([
            CrossModalBlock(
                embed_dim=hidden_dim*2,      # 512
                num_heads=xattn_heads,
                ff_mult=xattn_ff_mult,
                dropout=xattn_dropout
            )
            for _ in range(xattn_layers)
        ])

        # ===== 4) 分类头：Multimodal(512) + Social(256) -> 768 =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2 + hidden_dim, 256),  # 512 + 256 = 768 -> 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # （可选）打印参数统计
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MHCA层数: {xattn_layers}, 训练/总参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    def forward(self, bert_input_ids, bert_attention_mask,
                      roberta_input_ids, roberta_attention_mask,
                      vit_pixel_values, clip_pixel_values, social_features):

        # ---- 文本模态：BERT & RoBERTa ----
        bert_out = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
        rob_out  = self.roberta(input_ids=roberta_input_ids, attention_mask=roberta_attention_mask)
        t1 = self.bert_proj(bert_out.pooler_output)      # [B,256]
        t2 = self.roberta_proj(rob_out.pooler_output)    # [B,256]
        text = torch.cat([t1, t2], dim=1).unsqueeze(1)   # [B,1,512]

        # ---- 图像模态：ViT & CLIP-Visual ----
        vit_out  = self.vit(pixel_values=vit_pixel_values)
        clip_out = self.clip_visual(pixel_values=clip_pixel_values)
        i1 = self.vit_proj(vit_out.pooler_output)        # [B,256]
        i2 = self.clip_proj(clip_out.pooler_output)      # [B,256]
        image = torch.cat([i1, i2], dim=1).unsqueeze(1)  # [B,1,512]

        # ---- 堆叠 N 层双向 MHCA ----
        for blk in self.xblocks:
            text, image = blk(text, image)               # 都保持 [B,1,512]

        # 融合两个模态（平均，也可换成 concat + 线性）
        multimodal = (text.squeeze(1) + image.squeeze(1)) / 2.0  # [B,512]

        # ---- 社交特征 ----
        social = self.social_proj(social_features)        # [B,256]

        # ---- 拼接 + 分类 ----
        fused = torch.cat([multimodal, social], dim=1)    # [B,768]
        return self.classifier(fused)


# ============================================
# 模块 7. 数据加载器
# ============================================
train_dataset = WeiboDataset(train_df, bert_tokenizer, roberta_tokenizer, vit_processor, clip_processor)
val_dataset   = WeiboDataset(val_df, bert_tokenizer, roberta_tokenizer, vit_processor, clip_processor)
test_dataset  = WeiboDataset(test_df, bert_tokenizer, roberta_tokenizer, vit_processor, clip_processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ============================================
# 模块 8. 初始化模型
# ============================================
model = MultiModalModel(bert_model, roberta_model, vit_model, clip_visual).to(DEVICE)

# ============================================
# 模块 9. 训练与评估工具函数
# ============================================
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
            batch["bert_input_ids"].to(DEVICE),
            batch["bert_attention_mask"].to(DEVICE),
            batch["roberta_input_ids"].to(DEVICE),
            batch["roberta_attention_mask"].to(DEVICE),
            batch["vit_pixel_values"].to(DEVICE),
            batch["clip_pixel_values"].to(DEVICE),
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
                batch["bert_input_ids"].to(DEVICE),
                batch["bert_attention_mask"].to(DEVICE),
                batch["roberta_input_ids"].to(DEVICE),
                batch["roberta_attention_mask"].to(DEVICE),
                batch["vit_pixel_values"].to(DEVICE),
                batch["clip_pixel_values"].to(DEVICE),
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
# 模块 10. 训练主循环 (早停 + 日志)
# ============================================
EPOCHS = 30
LR = 2e-5
PATIENCE = 5

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_val_f1 = 0.0
patience_counter = 0

os.makedirs("model", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
log_path = os.path.join("tmp", "training_log.csv")

with open(log_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    headers = [
        "epoch",
        "train_loss", "train_acc", "train_macro_f1",
        "train_precision_0", "train_recall_0", "train_f1_0",
        "train_precision_1", "train_recall_1", "train_f1_1",
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

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        torch.save(model.state_dict(), os.path.join("model", "best_multimodal_model.pth"))
        print("✅ 最佳模型已保存")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ 早停")
            break

# ============================================
# 模块 11. 测试集评估
# ============================================
print("\n=== 在测试集上评估 ===")
model.load_state_dict(torch.load(os.path.join("model", "best_multimodal_model.pth")))
test_metrics = evaluate(model, test_loader, criterion)

print(f"测试集: Loss={test_metrics['loss']:.4f}, Acc={test_metrics['acc']:.4f}, Macro-F1={test_metrics['macro_f1']:.4f}")

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            batch["bert_input_ids"].to(DEVICE),
            batch["bert_attention_mask"].to(DEVICE),
            batch["roberta_input_ids"].to(DEVICE),
            batch["roberta_attention_mask"].to(DEVICE),
            batch["vit_pixel_values"].to(DEVICE),
            batch["clip_pixel_values"].to(DEVICE),
            batch["social_features"].to(DEVICE)
        )
        preds = outputs.argmax(1).cpu().numpy()
        labels = batch["label"].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

report = classification_report(all_labels, all_preds, target_names=["nonrumor(0)", "rumor(1)"], digits=4)
print(report)

cm = confusion_matrix(all_labels, all_preds)
print("\n混淆矩阵:\n", cm)
np.save(os.path.join("tmp", "confusion_matrix.npy"), cm)
