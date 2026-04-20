# 🤟 Video-Based Sign Language Recognition Model

<div align="center">

![Sign Language Recognition](https://img.shields.io/badge/Task-Sign%20Language%20Recognition-teal?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Approach-Deep%20Learning-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-71.63%25-brightgreen?style=for-the-badge)
![Classes](https://img.shields.io/badge/Classes-157%20Sentences-orange?style=for-the-badge)

<br/>

> *An end-to-end deep learning pipeline for continuous sign language sentence recognition using CNN–BiLSTM architecture.*

<br/>

[![Dataset](https://img.shields.io/badge/📦%20Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/belovedorange/nlp-dataset)
[![GitHub](https://img.shields.io/badge/💻%20Code-GitHub-181717?style=flat-square&logo=github)](https://github.com)

</div>

---

## 👥 Group Details

| Name | Roll Number |
|------|-------------|
| Alapati Saahitthi | 2022 IMT 011 |
| Diwakar Dubey | 2022 IMT 039 |
| Drishti Aggarwal | 2022 IMT 040 |
| Priya Keshri | 2022 IMT 061 |

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Data Preparation](#-data-preparation)
- [Model Architecture](#-model-architecture)
- [Training Setup](#-training-setup)
- [Results & Evaluation](#-results--evaluation)
- [Discussion & Limitations](#-discussion--limitations)
- [Future Improvements](#-future-improvements)
- [Conclusion](#-conclusion)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)

---

## 🧠 Introduction

### Problem Statement

Sign language communication is largely **inaccessible in digital systems** due to a lack of effective translation mechanisms. Most interfaces rely on text or speech, creating a significant barrier for deaf and hard-of-hearing individuals who use sign language as their primary mode of communication.

The core challenge is to **automatically recognize and classify sign language gestures from video data** into meaningful text. This is inherently complex because sign language involves:

- 🖐️ **Dynamic hand movements** — spatially rich, highly variable
- 😐 **Facial expressions** — carry grammatical meaning
- ⏱️ **Temporal context** — gestures evolve over time

### Objective

This project develops an **end-to-end deep learning pipeline** to process sign language videos and predict corresponding sentence-level labels. Specifically, the model:

1. **Extracts spatial features** from individual video frames using a pre-trained CNN (ResNet-50)
2. **Captures temporal dependencies** across frame sequences using an LSTM/BiLSTM
3. **Classifies each video** into one of 157 predefined sign language sentence categories

> 🎯 **Goal**: Not just accuracy, but a *scalable, robust pipeline* that handles real-world video inconsistencies and generalizes across different signers — laying the groundwork for real-time AI-driven sign language translation.

---

## 📦 Dataset

> **Dataset created by the project team and publicly available on Kaggle.**

[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/belovedorange/nlp-dataset)

| Property | Details |
|----------|---------|
| **Source** | Custom-built & published on Kaggle |
| **Format** | Video files (`.mp4`, `.avi`, `.mov`) |
| **Classes** | 157 sentence-level sign language labels |
| **Organization** | Folder-based — innermost folder name = gesture label |
| **Samples** | Multiple video files per class |

---

## 🗂️ Data Preparation

### Video Loading

Videos are loaded using a custom **PyTorch `Dataset` class** that:

- Traverses a hierarchical folder structure to locate video files
- Uses **OpenCV (`cv2.VideoCapture`)** for frame reading
- Converts frames from **BGR → RGB** (aligning with deep learning conventions)
- Replaces unreadable/corrupt videos with **zero tensors** to ensure uninterrupted training

### Frame Extraction Strategy

To handle videos of varying durations, a **fixed-length frame sampling** approach is used:

```
num_frames = 32 (improved model)  /  20 (baseline)
```

| Scenario | Strategy |
|----------|----------|
| Video has **more** frames than required | Uniform sampling via `numpy.linspace` — evenly spaced indices across the full video |
| Video has **fewer** frames than required | Last frame is repeated to pad to required length |

This guarantees every sample is a consistent tensor of shape:

```
(num_frames, 3, 224, 224)
```

### Preprocessing Pipeline

Each frame undergoes the following transformations:

```
Raw Frame (BGR)
     │
     ▼
BGR → RGB Conversion
     │
     ▼
Resize to 224×224 pixels
     │
     ▼
ToTensor() → Pixel values scaled to [0, 1]
     │
     ▼
ImageNet Normalization  ← (added in improved model)
     │
     ▼
Final Tensor: (3, 224, 224)
```

> **Note:** The baseline model lacked ImageNet normalization — adding it was one of the biggest contributors to performance improvement.

### Dataset Split

```
Total Dataset
├── Training Set (80%)   → learns model parameters
└── Validation Set (20%) → evaluates generalization
```

- Implemented using PyTorch's `random_split`
- Both subsets share the same label mapping and preprocessing logic
- No explicit test set at this stage; validation set serves as a performance proxy

---

## 🏗️ Model Architecture

### Overview

The model uses a **hybrid CNN–BiLSTM architecture** designed to separate and jointly optimize spatial and temporal learning:

```
Input Video
     │
     ▼
┌─────────────────────────────┐
│   Frame Sampling            │  Uniform / Padding → Fixed T frames
│   (Uniform / Padding)       │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Preprocessing             │  Resize 224×224, Normalize, RGB
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Frame-wise Flattening     │  (B × T, 3, 224, 224)
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   CNN Backbone              │  ResNet-50 (frozen weights)
│   (ResNet-50)               │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Feature Extraction        │  (B × T, 2048)
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Reshape to Sequence       │  (B, T, 2048)
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   BiLSTM                    │  Bidirectional LSTM, hidden=256
│   (Temporal Modeling)       │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Final Hidden State        │  (B, 256)
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Fully Connected Layer     │  256 → num_classes (157)
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Softmax Output            │  Class probabilities
└─────────────────────────────┘
```

### Component Details

| Component | Description | Output Shape |
|-----------|-------------|-------------|
| **Input Video** | Raw video clip as a frame sequence | `(B, T, H, W, C)` |
| **Frame Sampling** | Uniform or padded to fixed length T | `(B, T, H, W, C)` |
| **Preprocessing** | Resize, normalize, BGR→RGB | `(B, T, 3, 224, 224)` |
| **Frame-wise Flattening** | Reshape for independent CNN processing | `(B×T, 3, 224, 224)` |
| **CNN (ResNet-50)** | Spatial feature extractor, final FC removed | `(B×T, 2048)` |
| **Reshape to Sequence** | Restore temporal structure | `(B, T, 2048)` |
| **BiLSTM** | Bidirectional temporal modeling | `(B, 256)` |
| **Final Hidden State** | Compact sequence-level representation | `(B, 256)` |
| **Fully Connected** | Maps representation to class logits | `(B, 157)` |
| **Softmax Output** | Probability distribution over classes | `(B, 157)` |

### Why This Architecture?

| Consideration | Reasoning |
|---------------|-----------|
| **ResNet-50 (frozen)** | Pre-trained on ImageNet — rich visual features without large-scale retraining; reduces training time |
| **BiLSTM over LSTM** | Processes frames in both forward and backward directions; captures full gesture context |
| **CNN–LSTM over 3D CNN** | More memory-efficient; avoids heavy spatiotemporal convolutions |
| **Fixed frame sampling** | Handles variable-length videos while maintaining consistent input dimensions for batch processing |

---

## ⚙️ Training Setup

| Component | Configuration |
|-----------|--------------|
| **Loss Function** | Cross-Entropy Loss (multi-class classification) |
| **Optimizer** | AdamW with weight decay (regularization) |
| **Batch Size** | 4 (high memory cost per sample due to multi-frame input) |
| **Epochs** | Up to 15 (early stopping applied) |
| **Learning Rate Schedule** | Reduced on validation plateau |
| **Early Stopping** | Patience = 3 epochs |
| **Gradient Clipping** | Applied for stable convergence |
| **Hardware** | GPU-enabled Kaggle environment (fallback: CPU) |
| **Framework** | PyTorch + OpenCV + torchvision |

---

## 📊 Results & Evaluation

### Metrics

- **Accuracy** — Primary metric: ratio of correct predictions to total predictions
- **Cross-Entropy Loss** — Monitors divergence between predicted probabilities and ground truth
- **Early Stopping** — Saves model weights at peak validation accuracy

### Final Model Performance

| Metric | Value |
|--------|-------|
| 🏋️ Training Accuracy | **76.21%** |
| ✅ Validation Accuracy | **71.63%** |
| 📉 Training Loss | **0.6852** |
| 🔢 Classes | 157 |
| 📅 Peak Epoch | Epoch 14 |

### Baseline vs. Improved Model

```
Accuracy (%)
│
80 │                                          ▓▓▓ 76.21%
   │                                        ▓▓
   │                                      ▓▓
   │                                    ▓▓     ░░░ 71.63%
   │                                  ▓▓     ░░
60 │                              ▓▓▓▓     ░░
   │                            ▓▓       ░░
   │                          ▓▓       ░░
   │                        ▓▓       ░░
40 │                      ▓▓       ░░
   │                    ▓▓       ░░
   │                  ▓▓       ░░
   │               ▓▓▓       ░░
20 │             ▓▓         ░░
   │           ▓▓          ░░
   │        ▓▓▓          ░░
   │      ▓▓            ░░
 0 │▓▓▓▓▓░░░░░────────────────────────────────────
   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
                              Epochs
              ▓ Training Accuracy    ░ Validation Accuracy
```

| Model Version | Training Accuracy | Validation Accuracy |
|---------------|-------------------|---------------------|
| **Version 1 (Baseline)** | ~5–6% | ~7% |
| **Version 2 (Improved)** | **76.21%** | **71.63%** |

> 📌 Both training and validation accuracy improve steadily up to **Epoch 14**, after which validation accuracy declines while training accuracy continues to rise — a clear sign of overfitting, mitigated by early stopping.

### Key Improvements from Baseline → Final

| Change | Impact |
|--------|--------|
| Added **ImageNet normalization** | Better spatial feature extraction from frozen ResNet-50 |
| Increased **frame count** (20 → 32) | Richer temporal coverage of each gesture |
| Switched **LSTM → BiLSTM** | Model reads sequences forwards and backwards for fuller context |
| Applied **AdamW + weight decay** | Better regularization, reduced overfitting |
| Added **dropout** | Further variance control |
| **Learning rate scheduling** | More stable and efficient convergence |

### Qualitative Sample Predictions

#### ✅ Correct Predictions
| Actual | Predicted | Result |
|--------|-----------|--------|
| `how can i help you` | `how can i help you` | ✅ |
| `i am bored` | `i am bored` | ✅ |

#### ⚠️ Confused Predictions (Similar Gestures)
| Actual | Predicted | Reason |
|--------|-----------|--------|
| `how can i help you` | `can you help me` | Overlapping motion patterns |
| `pour some more water in the glass` | `bring water for me` | Similar hand movements |

#### ❌ Incorrect Predictions
| Actual | Predicted | Reason |
|--------|-----------|--------|
| `i am suffering from fever` | `i am bored` | Fast/unclear gestures |
| `you are bad` | `how can i help you` | Insufficient class samples |

---

## 🔍 Discussion & Limitations

### Observations

- 📈 **Preprocessing quality** was as critical as model architecture — normalization alone drove major gains
- 🔁 **BiLSTM** outperformed LSTM by capturing both past and future frame context
- ⏹️ **Early stopping** was essential — overfitting emerged clearly after Epoch 14
- 🌍 Model is sensitive to **signing speed**, **individual style**, and **environmental conditions**

### Limitations

| Limitation | Description |
|------------|-------------|
| **Limited data per class** | Small per-class sample count reduces fine-grained discrimination |
| **High class count (157)** | Many classes share visually similar gestures |
| **Frozen CNN backbone** | ResNet-50 features are generic (ImageNet) — not adapted to gesture-specific patterns |
| **No keypoint extraction** | No hand landmarks, finger-level motion, or optical flow |
| **Fixed-length sampling** | Rapid gesture transitions may be missed during uniform frame selection |
| **Closed-set classification** | Cannot recognize gestures outside the 157 training classes |
| **Sensitivity to variations** | Signing speed, style differences, lighting, and background affect performance |
| **Computational cost** | ResNet-50 + BiLSTM is resource-intensive; limits real-time scalability |

---

## 🚀 Future Improvements

### Short-Term

- [ ] **Fine-tune CNN backbone** — Unfreeze later ResNet-50 layers on CSL data for gesture-specific spatial features
- [ ] **Expand dataset** — Add more videos per class with diverse signers, speeds, and environments
- [ ] **Add attention layers** — Focus model on the most informative frames in a sequence

### Medium-Term

- [ ] **Integrate hand/pose keypoints** — Use MediaPipe to extract explicit landmarks (finger joints, body pose)
- [ ] **Incorporate optical flow** — Explicitly model motion vectors between consecutive frames
- [ ] **Adaptive frame sampling** — Keyframe-based selection instead of uniform sampling

### Long-Term

- [ ] **Transformer-based models** — Replace BiLSTM with Vision Transformers (ViT) or Temporal Transformers for long-range dependencies
- [ ] **3D CNNs** — Explore spatiotemporal convolutions (e.g., SlowFast, I3D)
- [ ] **Class balancing** — Oversampling + weighted loss functions for equitable learning across all 157 classes
- [ ] **Real-time optimization** — Lighter backbones (MobileNet, ResNet-18) for deployment
- [ ] **Open-set recognition** — Move beyond closed-set classification to handle unseen gestures

---

## 🏁 Conclusion

This project demonstrates that **effective Continuous Sign Language (CSL) recognition is not achieved through model selection alone**, but through the systematic co-optimization of:

1. 🔧 Data quality and preprocessing
2. 🏗️ Architecture design (CNN + BiLSTM)
3. 📈 Training strategy (AdamW, LR scheduling, early stopping)

Starting from a near-random baseline (~7%), the final model achieves **71.63% validation accuracy** across **157 sentence classes** — a result that underscores the value of an end-to-end pipeline perspective.

The work serves as a **foundational step** toward real-time sign language translation systems that can bridge communication gaps between sign language users and the broader world.

---

## 📁 Project Structure

```
📦 sign-language-recognition/
├── 📂 data/
│   └── (video files organized by label folders)
├── 📂 models/
│   ├── cnn_lstm_model.py        # CNN-BiLSTM architecture definition
│   └── checkpoints/             # Saved model weights
├── 📂 notebooks/
│   ├── data_exploration.ipynb   # Dataset analysis
│   └── training.ipynb           # Training pipeline
├── 📂 src/
│   ├── dataset.py               # PyTorch Dataset class (video loading + preprocessing)
│   ├── train.py                 # Training loop with early stopping
│   ├── evaluate.py              # Evaluation and metrics
│   └── predict.py               # Inference on new videos
├── 📂 results/
│   └── training_curves.png      # Training vs Validation accuracy plots
├── requirements.txt
└── README.md
```

---

## 🛠️ Getting Started

### Prerequisites

```bash
pip install torch torchvision opencv-python numpy
```

### Dataset

Download the dataset from Kaggle:

```bash
# Using Kaggle CLI
kaggle datasets download -d belovedorange/nlp-dataset
```

Or directly from: [https://www.kaggle.com/datasets/belovedorange/nlp-dataset](https://www.kaggle.com/datasets/belovedorange/nlp-dataset)

### Training

```python
# Basic training run
python src/train.py \
  --data_dir ./data \
  --num_frames 32 \
  --batch_size 4 \
  --epochs 20 \
  --early_stopping_patience 3
```

### Inference

```python
from src.predict import predict_video

result = predict_video(
    video_path="path/to/video.mp4",
    model_checkpoint="models/checkpoints/best_model.pth"
)
print(f"Predicted: {result}")
```

---

## 📚 Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)

---

<div align="center">

Made with ❤️ by the NLP Project Team — IIT Roorkee IMT 2022

</div>
