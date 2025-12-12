Weibo-Multimodal-Rumor-Detection

Data Processing and Preprocessing Pipeline

This project uses the publicly released Weibo multimodal rumor detection dataset published by Wang et al.
The original dataset and download links can be found at:
👉 https://github.com/wangzhuang1911/Weibo-dataset
The preprocessing pipeline aligns text, images, and social features into a unified and reproducible format suitable for multimodal learning.

🔹 1. Download the Original Dataset
Clone or download the official repository:
git clone https://github.com/wangzhuang1911/Weibo-dataset
The repository contains:
train_nonrumor.txt
train_rumor.txt
test_nonrumor.txt
test_rumor.txt
social_feature.txt
Image folders with multiple images per tweet
These files include text, labels, raw tweet IDs, image references, and 16-dim social statistical features.

🔹 2. Restore Tweet IDs and Collect Raw Text Content
The original TXT files store Weibo posts indexed by weibo_id.
Because some IDs exceed 32-bit integer range, Excel/CSV tools may convert them into scientific notation, causing mapping errors.
To avoid this:
Extract weibo_id directly from filenames and TXT content
Keep all IDs as strings to prevent precision loss
Parse each TXT file to extract:
Tweet ID
Text content
Associated image names
Rumor / non-rumor label
This ensures lossless reconstruction of the original mapping.

🔹 3. Download and Align Image Files
Image folders contain multiple images for each tweet.
The pipeline:
Matches images to posts using filename patterns (e.g., 1234567890_1.jpg)
For posts with multiple images, all image paths are recorded
During model preprocessing, multiple images are averaged into a single representative image (as used in the paper)
This step ensures one-to-one alignment between text and visual content.

🔹 4. Integrate Social Features
The file social_feature.txt provides a 16-dimensional numeric vector per tweet, including:
repost count
comment count
like count
user influence
interaction indicators
account activity statistics
Preprocessing steps:
Load the 16-d vector for each weibo_id
Align with text + image entries
Remove samples with missing or inconsistent records
Produce a complete multimodal item:
{weibo_id, text, image_files, social_features, label}

🔹 5. Final Dataset Construction
After text, image, and social alignment, the data are merged into structured CSV files:
train_rumor.csv
train_nonrumor.csv
test_rumor.csv
test_nonrumor.csv
social_feature.csv
Each row contains:
weibo_id | text | image_list | social_vector | label
This structure is fully compatible with the models in this repository.

🔹 6. Train / Validation / Test Split
The original Weibo dataset provides only train and test subsets.
To enable proper model selection, we follow the setting used in the paper:
From the original training set, we sample 10% as a validation set
The resulting split is approximately:
70% training
10% validation
20% test
Random seeds are fixed to ensure reproducibility.

🔹 7. Reproducibility and Availability
All preprocessing code—including:
TXT parsing
weibo_id restoration
text–image–social alignment
CSV generation
…is released in this repository so that others can fully reproduce our processed dataset.

⚠️ Due to copyright and licensing restrictions, we do not redistribute the original Weibo posts or images.
Users must download the raw dataset from the official source.




Model Descriptions

This repository contains nine progressively enhanced multimodal rumor detection models.
Each model introduces new components or fusion strategies, moving from simple frozen baselines to the final gated and robust fusion architecture.

🔹 model1.py — Three-Modality Frozen Baseline
Implements a three-modality architecture:
text semantic (BERT) + image semantic (ViT) + social features.
All pretrained encoders are fully frozen.
Fusion uses simple concatenation followed by an MLP classifier.
Serves as the minimal and most lightweight baseline.

🔹 model2.py — Three-Modality with Partial Unfreezing
Extends model1 by enabling partial unfreezing of the top layers of BERT and ViT.
Fusion remains concatenation-based.
Prints parameter statistics for ablation comparisons.

🔹 model3.py — Standardized Three-Modality Frozen Implementation
Provides a cleaner, more complete version of the three-modality pipeline.
Includes improved preprocessing and alignment.
Encoders remain frozen; fusion still uses projection + concatenation.
Used as the standard frozen baseline for comparison.

🔹 model4.py — Refined Three-Modality with Partial Unfreezing
Builds upon model3 while incorporating controlled partial unfreezing.
Retains the three-branch concatenation architecture.
Evaluates parameter–performance trade-offs.

🔹 model5.py — Five-Branch Frozen Multimodal Baseline
Expands to five branches:
Text semantic (BERT)
Text affective cues (RoBERTa)
Image semantic (ViT)
Image affective cues (CLIPvision)
Social features
All pretrained encoders are frozen.
Multimodal fusion is direct concatenation of five projected vectors.
Represents the frozen version of the full five-branch architecture.

🔹 model6.py — Five-Branch with Partial Unfreezing
Extends model5 with partial fine-tuning of BERT, RoBERTa, ViT, and CLIPvision.
Fusion remains concatenation-based.
Used to study the effect of limited task-specific adaptation across multiple backbones.

🔹 model7.py — Cross-Attention Enhanced Multimodal Model
Introduces bidirectional cross-modal attention between text and image features.
Semantic and emotional features are merged within each modality prior to interaction.
The cross-attended representation is fused with social features.
Explicitly models text–image alignment beyond simple concatenation.

🔹 model8.py — Cross-Attention with Lightweight Gating
Builds on model7 by adding lightweight gating / weighting mechanisms.
Represents a hybrid design combining cross-modal attention and initial gating control.
Used to explore joint effects of interaction and modality weighting.

🔹 model9.py — Final Gated and Robust Multimodal Fusion Model
This is the final version corresponding to the paper.
Implements the complete five-branch architecture:
BERT (semantic), RoBERTa (affective), ViT (semantic), CLIPvision (affective), and social features.
All branches are projected to a unified 256-dimensional latent space.
Core components include:
Modality-wise Sigmoid gating (dynamic weighting)
Modality-level dropout (robustness to missing/noisy modalities)
Concatenation → lightweight MLP classifier
This is the final, official, and recommended model for experiments.
