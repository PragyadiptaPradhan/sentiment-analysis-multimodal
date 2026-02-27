# models.py — Complete Line-by-Line Documentation with Examples

> This guide walks through every single line, class, and method in `models.py`, explaining what it does, why it's needed, and providing concrete examples with tensor shapes throughout.

---

## Table of Contents

- [1. Imports](#1-imports)
- [2. TextEncoder Class](#2-textencoder-class)
  - [2a. `__init__`](#2a-__init__)
  - [2b. `forward`](#2b-forward)
- [3. VideoEncoder Class](#3-videoencoder-class)
  - [3a. `__init__`](#3a-__init__)
  - [3b. `forward`](#3b-forward)
- [4. AudioEncoder Class](#4-audioencoder-class)
  - [4a. `__init__`](#4a-__init__)
  - [4b. `forward`](#4b-forward)
- [5. MultimodalSentimentModel Class](#5-multimodalsentimentmodel-class)
  - [5a. `__init__`](#5a-__init__)
  - [5b. `forward`](#5b-forward)
- [6. MultimodalTrainer Class](#6-multimodaltrainer-class)
  - [6a. `__init__`](#6a-__init__)
  - [6b. `log_metrics`](#6b-log_metrics)
  - [6c. `train_epoch`](#6c-train_epoch)
  - [6d. `evaluate`](#6d-evaluate)
- [7. Main Block (Testing Script)](#7-main-block-testing-script)
- [8. Full Architecture Diagram](#8-full-architecture-diagram)
- [9. Tensor Shape Cheat Sheet](#9-tensor-shape-cheat-sheet)

---

## 1. Imports

```python
import os
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import torch
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from meld_dataset import MELDDataset
```

| # | Import | Library | Purpose | Where Used |
|---|--------|---------|---------|------------|
| 1 | `os` | Standard Library | Check environment variables for SageMaker detection | `'SM_MODEL_DIR' in os.environ` in Trainer |
| 2 | `torch.nn as nn` | PyTorch | Neural network building blocks (layers, loss functions) | Every class inherits `nn.Module`; uses `nn.Linear`, `nn.Conv1d`, etc. |
| 3 | `BertModel` | HuggingFace Transformers | Pre-trained BERT language model | `TextEncoder` loads `bert-base-uncased` |
| 4 | `vision_models` | torchvision | Pre-trained video models | `VideoEncoder` loads `r3d_18` (3D ResNet-18) |
| 5 | `torch` | PyTorch | Core tensor operations, optimizers, device management | `torch.cat`, `torch.optim.Adam`, `torch.inference_mode` |
| 6 | `precision_score, accuracy_score` | scikit-learn | Classification metrics computation | `evaluate()` computes precision and accuracy |
| 7 | `SummaryWriter` | TensorBoard | Writes training metrics for visualization | `log_metrics()` writes scalars to TensorBoard |
| 8 | `datetime` | Standard Library | Generate timestamps for log directories | `datetime.now().strftime('%b%d_%H-%M-%S')` |
| 9 | `MELDDataset` | Local module | Dataset class for loading MELD data | Used only in `__main__` block for testing |

---

## 2. TextEncoder Class

**Purpose:** Encode text utterances into 128-dimensional feature vectors using pre-trained BERT.

```python
class TextEncoder(nn.Module):
```

- Inherits from `nn.Module` — PyTorch base class for all neural network modules
- Enables `.parameters()`, `.to(device)`, `.train()`, `.eval()`, and automatic gradient tracking

---

### 2a. `__init__`

```python
def __init__(self):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')

    for param in self.bert.parameters():
        param.requires_grad = False

    self.projection = nn.Linear(768, 128)
```

| # | Code | Explanation |
|---|------|-------------|
| 1 | `super().__init__()` | Calls `nn.Module.__init__()` — required to register layers, parameters, and hooks |
| 2 | `BertModel.from_pretrained('bert-base-uncased')` | Downloads and loads pre-trained BERT model (110M parameters, 12 layers, 768 hidden size, 30K vocab, lowercase) |
| 3 | `for param in self.bert.parameters():` | Iterates over all ~110M BERT parameters |
| 4 | `param.requires_grad = False` | **Freezes BERT** — no gradients computed, weights don't update during training |
| 5 | `nn.Linear(768, 128)` | Linear projection layer: maps BERT's 768-dim output → 128-dim feature vector. This layer IS trainable |

**Why freeze BERT?**
- BERT is already pre-trained on massive text corpora
- Fine-tuning all 110M parameters would require huge GPU memory and risk overfitting on MELD's ~10K samples
- Only the projection layer (768×128 + 128 bias = 98,432 params) is trained

**Example:**
```python
encoder = TextEncoder()

# Count parameters:
total_params = sum(p.numel() for p in encoder.parameters())
trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"Total: {total_params:,}")       # Total: 109,580,544
print(f"Trainable: {trainable_params:,}")  # Trainable: 98,432
```

### What `nn.Linear(768, 128)` does mathematically:

$$\text{output} = \mathbf{x} \cdot \mathbf{W}^T + \mathbf{b}$$

Where:
- $\mathbf{x}$: input tensor of shape `(batch_size, 768)`
- $\mathbf{W}$: weight matrix of shape `(128, 768)`
- $\mathbf{b}$: bias vector of shape `(128,)`
- output: tensor of shape `(batch_size, 128)`

---

### 2b. `forward`

```python
def forward(self, input_ids, attention_mask):
    # Extract BERT features
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

    # Use the [CLS] token representation
    pooled_output = outputs.pooler_output

    return self.projection(pooled_output)
```

| # | Code | Input Shape | Output Shape | Explanation |
|---|------|------------|-------------|-------------|
| 1 | `self.bert(input_ids, attention_mask)` | `(B, 128)`, `(B, 128)` | `BaseModelOutputWithPooling` | Run BERT forward pass on tokenized text |
| 2 | `outputs.pooler_output` | — | `(B, 768)` | Extract the [CLS] token's hidden state, passed through a tanh activation — represents the entire sentence |
| 3 | `self.projection(pooled_output)` | `(B, 768)` | `(B, 128)` | Project 768-dim BERT output to 128-dim feature space |

**Example step-by-step (batch_size=4):**
```python
# Input
input_ids.shape      # (4, 128) — 4 sentences, each 128 tokens
attention_mask.shape  # (4, 128) — 1 for real tokens, 0 for padding

# After BERT
outputs = self.bert(input_ids, attention_mask)
outputs.last_hidden_state.shape  # (4, 128, 768) — all token embeddings
outputs.pooler_output.shape      # (4, 768) — [CLS] token embedding

# After projection
result = self.projection(outputs.pooler_output)
result.shape  # (4, 128) — compressed text features
```

**What is `pooler_output`?**
```
Input tokens:  [CLS] I ' m   so  happy ! [SEP] [PAD] [PAD] ...
                 ↓
BERT processes all tokens through 12 transformer layers
                 ↓
pooler_output = BERT's hidden state at position 0 ([CLS]) → Linear → Tanh
                 ↓
Shape: (batch_size, 768) — one vector per sentence
```

---

## 3. VideoEncoder Class

**Purpose:** Encode video clips (30 frames) into 128-dimensional feature vectors using pre-trained R3D-18.

---

### 3a. `__init__`

```python
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
```

| # | Code | Explanation |
|---|------|-------------|
| 1 | `super().__init__()` | Initialize `nn.Module` base class |
| 2 | `vision_models.video.r3d_18(pretrained=True)` | Load pre-trained R3D-18 (3D ResNet-18 for video understanding, trained on Kinetics-400 dataset). Uses 3D convolutions over space + time |
| 3 | `for param ... requires_grad = False` | **Freeze all R3D-18 parameters** (~33M params) — use as fixed feature extractor |
| 4 | `num_ftrs = self.backbone.fc.in_features` | Get the input size of R3D-18's original final FC layer (512) |
| 5 | `self.backbone.fc = nn.Sequential(...)` | **Replace** the original FC layer with a custom head |
| 6 | `nn.Linear(num_ftrs, 128)` | 512 → 128 linear projection |
| 7 | `nn.ReLU()` | Rectified Linear Unit activation: $f(x) = \max(0, x)$ — adds nonlinearity |
| 8 | `nn.Dropout(0.2)` | Randomly zeros 20% of neurons during training to prevent overfitting |

**What is R3D-18?**
```
R3D-18 = ResNet-18 but with 3D convolutions
- Normal ResNet: 2D conv (height × width) — for images
- R3D-18: 3D conv (time × height × width) — for videos

Architecture:
Input → Conv3D → BatchNorm3D → ReLU → MaxPool3D
     → ResBlock×2 → ResBlock×2 → ResBlock×2 → ResBlock×2
     → AdaptiveAvgPool3D → FC(512 → 400)
                           ↑
                    We REPLACE this with:
                    FC(512 → 128) → ReLU → Dropout(0.2)
```

**Example:**
```python
encoder = VideoEncoder()

# Original R3D-18 FC layer:
# nn.Linear(512, 400) — 400 Kinetics action classes

# After replacement:
# nn.Sequential(
#     nn.Linear(512, 128),
#     nn.ReLU(),
#     nn.Dropout(0.2)
# )

# Parameter count:
total = sum(p.numel() for p in encoder.parameters())
trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"Total: {total:,}")       # ~33,300,000
print(f"Trainable: {trainable:,}")  # 65,664 (512*128 + 128 bias)
```

---

### 3b. `forward`

```python
def forward(self, x):
    # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
    x = x.transpose(1, 2)
    return self.backbone(x)
```

| # | Code | Input Shape | Output Shape | Explanation |
|---|------|------------|-------------|-------------|
| 1 | `x.transpose(1, 2)` | `(B, 30, 3, 224, 224)` | `(B, 3, 30, 224, 224)` | Swap frames and channels dimensions — R3D-18 expects channels first |
| 2 | `self.backbone(x)` | `(B, 3, 30, 224, 224)` | `(B, 128)` | Run through entire R3D-18 + custom FC head |

**Why `.transpose(1, 2)`?**
```python
# MELDDataset outputs: [batch, frames, channels, height, width]
# Input shape:          (4,     30,     3,       224,    224)
#                             ↑ dim1   ↑ dim2

# R3D-18 expects:      [batch, channels, frames, height, width]
# After transpose:      (4,     3,       30,     224,    224)
#                              ↑ dim1    ↑ dim2   (swapped!)

x = x.transpose(1, 2)
# transpose(1, 2) swaps dimensions at index 1 and 2
```

**Full forward pass example:**
```python
# Video input: 4 clips, each 30 frames of 224×224 RGB
video = torch.randn(4, 30, 3, 224, 224)

encoder = VideoEncoder()
output = encoder(video)
print(output.shape)  # (4, 128)
# Each video clip → one 128-dim feature vector
```

---

## 4. AudioEncoder Class

**Purpose:** Encode mel spectrograms into 128-dimensional feature vectors using 1D convolutions.

---

### 4a. `__init__`

```python
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1d = nn.Sequential(
            # Lower Level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        for param in self.Conv1d.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
```

#### Conv1d Block — Layer by Layer

| # | Layer | Input Shape | Output Shape | What It Does |
|---|-------|------------|-------------|-------------|
| 1 | `nn.Conv1d(64, 64, kernel_size=3)` | `(B, 64, 300)` | `(B, 64, 298)` | Applies 64 1D convolution filters of size 3 across the time axis. Detects low-level patterns (phonemes, pitch changes). Output shrinks by `kernel_size - 1 = 2` |
| 2 | `nn.BatchNorm1d(64)` | `(B, 64, 298)` | `(B, 64, 298)` | Normalizes each channel to mean=0, std=1 across the batch. Stabilizes training |
| 3 | `nn.ReLU()` | `(B, 64, 298)` | `(B, 64, 298)` | $f(x) = \max(0, x)$ — zeroes out negative values |
| 4 | `nn.MaxPool1d(kernel_size=2)` | `(B, 64, 298)` | `(B, 64, 149)` | Takes max value in windows of 2 along time axis. Halves temporal resolution |
| 5 | `nn.Conv1d(64, 128, kernel_size=3)` | `(B, 64, 149)` | `(B, 128, 147)` | Expands to 128 channels. Detects higher-level audio patterns (words, intonation) |
| 6 | `nn.BatchNorm1d(128)` | `(B, 128, 147)` | `(B, 128, 147)` | Normalizes 128 channels |
| 7 | `nn.ReLU()` | `(B, 128, 147)` | `(B, 128, 147)` | Non-linear activation |
| 8 | `nn.AdaptiveAvgPool1d(1)` | `(B, 128, 147)` | `(B, 128, 1)` | Averages the ENTIRE time dimension into a single value per channel. Handles variable-length inputs |

#### Frozen + Projection

| Code | Explanation |
|------|-------------|
| `for param in self.Conv1d.parameters(): param.requires_grad = False` | Freeze conv layers (no gradient updates) |
| `nn.Linear(128, 128)` | 128 → 128 linear transformation |
| `nn.ReLU()` | Non-linear activation |
| `nn.Dropout(0.2)` | Drop 20% during training |

**What 1D Convolution does on audio:**
```
Mel Spectrogram input: 64 mel bands × 300 time steps
                       ↓
Imagine each mel band as a "channel" and time as the spatial dimension:

Channel 0 (low freq):  ▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁...  (300 values)
Channel 1:             ▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁▁▂...  (300 values)
...
Channel 63 (high freq): ▁▁▂▂▃▃▂▂▁▁▁▁▂▂▃▃▂▂▁▁▁▁▂▂▃▃...  (300 values)

Conv1d(kernel_size=3) slides a filter of width 3 across time:
[w1, w2, w3] × [val_t, val_t+1, val_t+2] = one output value
```

---

### 4b. `forward`

```python
def forward(self, x):
    x = x.squeeze(1)
    features = self.Conv1d(x)
    return self.projection(features.squeeze(-1))
```

| # | Code | Input Shape | Output Shape | Explanation |
|---|------|------------|-------------|-------------|
| 1 | `x.squeeze(1)` | `(B, 1, 64, 300)` | `(B, 64, 300)` | Removes channel dim (mono audio has 1 channel). Conv1d expects `(batch, channels, length)` |
| 2 | `self.Conv1d(x)` | `(B, 64, 300)` | `(B, 128, 1)` | Run through all conv + pool layers |
| 3 | `features.squeeze(-1)` | `(B, 128, 1)` | `(B, 128)` | Remove last dimension (size 1 from AdaptiveAvgPool) |
| 4 | `self.projection(...)` | `(B, 128)` | `(B, 128)` | Final linear projection with ReLU and dropout |

**Example:**
```python
encoder = AudioEncoder()

# Mel spectrogram: 4 samples, 1 channel, 64 mel bands, 300 time steps
audio = torch.randn(4, 1, 64, 300)

output = encoder(audio)
print(output.shape)  # (4, 128) — one 128-dim vector per audio clip
```

**What `.squeeze()` does:**
```python
# squeeze(dim) removes a dimension of size 1 at that position
x = torch.randn(4, 1, 64, 300)  # shape: (4, 1, 64, 300)
x.squeeze(1)                      # shape: (4, 64, 300) — removed dim 1

y = torch.randn(4, 128, 1)       # shape: (4, 128, 1)
y.squeeze(-1)                     # shape: (4, 128) — removed last dim
```

---

## 5. MultimodalSentimentModel Class

**Purpose:** Combines all three encoders, fuses their outputs, and classifies emotion (7 classes) and sentiment (3 classes).

---

### 5a. `__init__`

```python
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification
        self.emo_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 7)  # 7 emotions
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # 3 sentiments
        )
```

#### Encoders

| Encoder | Input | Output | Pre-trained On |
|---------|-------|--------|----------------|
| `TextEncoder()` | Token IDs + Attention Mask | `(B, 128)` | English Wikipedia + BookCorpus (BERT) |
| `VideoEncoder()` | 30 RGB frames 224×224 | `(B, 128)` | Kinetics-400 action dataset (R3D-18) |
| `AudioEncoder()` | Mel spectrogram 64×300 | `(B, 128)` | Not pre-trained (random init, but frozen) |

#### Fusion Layer — Line by Line

| # | Layer | Input Shape | Output Shape | What It Does |
|---|-------|------------|-------------|-------------|
| 1 | `nn.Linear(128 * 3, 256)` | `(B, 384)` | `(B, 256)` | Combines all 3 modalities (128+128+128=384) into 256-dim shared space |
| 2 | `nn.BatchNorm1d(256)` | `(B, 256)` | `(B, 256)` | Normalizes across batch for stable training |
| 3 | `nn.ReLU()` | `(B, 256)` | `(B, 256)` | Non-linear activation |
| 4 | `nn.Dropout(0.3)` | `(B, 256)` | `(B, 256)` | Drop 30% during training (higher than encoders because this is the fusion bottleneck) |

**Why 384 → 256?**
- 128 (text) + 128 (video) + 128 (audio) = 384 concatenated features
- Compressing to 256 forces the network to learn cross-modal interactions
- The fusion layer learns which modality combinations matter for each prediction

#### Emotion Classifier

| # | Layer | Input → Output | What It Does |
|---|-------|---------------|-------------|
| 1 | `nn.Linear(256, 128)` | `(B, 256)` → `(B, 128)` | Hidden layer |
| 2 | `nn.ReLU()` | — | Activation |
| 3 | `nn.Dropout(0.2)` | — | Regularization |
| 4 | `nn.Linear(128, 7)` | `(B, 128)` → `(B, 7)` | Output 7 raw logits (one per emotion class) |

**7 emotion classes:**
```
Index 0: anger    Index 1: disgust   Index 2: fear
Index 3: joy      Index 4: neutral   Index 5: sadness
Index 6: surprise
```

#### Sentiment Classifier

| # | Layer | Input → Output | What It Does |
|---|-------|---------------|-------------|
| 1 | `nn.Linear(256, 128)` | `(B, 256)` → `(B, 128)` | Hidden layer |
| 2 | `nn.ReLU()` | — | Activation |
| 3 | `nn.Dropout(0.2)` | — | Regularization |
| 4 | `nn.Linear(128, 3)` | `(B, 128)` → `(B, 3)` | Output 3 raw logits (one per sentiment class) |

**3 sentiment classes:**
```
Index 0: negative    Index 1: neutral    Index 2: positive
```

**Note:** The output is raw logits (unnormalized scores), NOT probabilities. `CrossEntropyLoss` internally applies softmax.

---

### 5b. `forward`

```python
def forward(self, text_inputs, video_frames, audio_features):
    text_features = self.text_encoder(
        text_inputs['input_ids'],
        text_inputs['attention_mask'],
    )
    video_features = self.video_encoder(video_frames)
    audio_features = self.audio_encoder(audio_features)

    # Concatenate multimodal features
    combined_features = torch.cat([
        text_features,
        video_features,
        audio_features
    ], dim=1)  # [batch_size, 128 * 3]

    # Fusion layer
    fused_features = self.fusion_layer(combined_features)

    emotion_output = self.emo_classifier(fused_features)
    sentiment_output = self.sentiment_classifier(fused_features)

    return {
        'emotion': emotion_output,
        'sentiment': sentiment_output
    }
```

#### Line-by-Line with Shapes (batch_size=16)

| # | Code | Shape | Explanation |
|---|------|-------|-------------|
| 1 | `self.text_encoder(input_ids, attention_mask)` | `(16, 128)` | Encode text → 128-dim vectors |
| 2 | `self.video_encoder(video_frames)` | `(16, 128)` | Encode video → 128-dim vectors |
| 3 | `self.audio_encoder(audio_features)` | `(16, 128)` | Encode audio → 128-dim vectors |
| 4 | `torch.cat([text, video, audio], dim=1)` | `(16, 384)` | Concatenate along feature dimension |
| 5 | `self.fusion_layer(combined_features)` | `(16, 256)` | Fuse modalities into shared representation |
| 6 | `self.emo_classifier(fused_features)` | `(16, 7)` | Produce emotion logits |
| 7 | `self.sentiment_classifier(fused_features)` | `(16, 3)` | Produce sentiment logits |

**What `torch.cat` does:**
```python
text_features  = torch.randn(16, 128)  # [f_t1, f_t2, ..., f_t128]
video_features = torch.randn(16, 128)  # [f_v1, f_v2, ..., f_v128]
audio_features = torch.randn(16, 128)  # [f_a1, f_a2, ..., f_a128]

combined = torch.cat([text_features, video_features, audio_features], dim=1)
# combined.shape = (16, 384)
# Each row: [f_t1, ..., f_t128, f_v1, ..., f_v128, f_a1, ..., f_a128]
#            └─── text 128 ───┘ └─── video 128 ───┘ └─── audio 128 ───┘
```

**Return value example:**
```python
output = model(text_inputs, video_frames, audio_features)

output['emotion'].shape     # (16, 7) — raw logits
output['sentiment'].shape   # (16, 3) — raw logits

# Example values (not probabilities!):
output['emotion'][0]   # tensor([-0.23, 0.15, -0.87, 1.42, 0.31, -0.56, 0.08])
#                                anger  disg   fear   joy   neut   sad    surp
# Highest logit = joy (1.42) → predicted emotion: joy

output['sentiment'][0]  # tensor([-0.45, 0.12, 0.89])
#                                 neg    neut   pos
# Highest logit = positive (0.89) → predicted sentiment: positive
```

---

## 6. MultimodalTrainer Class

**Purpose:** Manages the entire training and evaluation pipeline: optimizer, scheduler, loss computation, metric logging, and TensorBoard integration.

**Note:** This is NOT an `nn.Module` — it's a plain Python class. It wraps the model and handles the training loop.

---

### 6a. `__init__`

```python
class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `self.model = model` | Store reference to `MultimodalSentimentModel` |
| 2 | `self.train_loader = train_loader` | Store training DataLoader |
| 3 | `self.val_loader = val_loader` | Store validation DataLoader |

#### Dataset Size Logging

```python
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset Sizes: ")
        print(f"Train samples: {train_size} samples")
        print(f"Validation samples: {val_size} samples\n")
        print(f"Batch and epochs: {len(train_loader):,}")
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `len(train_loader.dataset)` | Total number of samples in training set |
| 2 | `len(val_loader.dataset)` | Total number of samples in validation set |
| 3 | `len(train_loader)` | Number of batches per epoch (samples / batch_size) |
| 4 | `{:,}` | Format with comma separator (e.g., `1,249`) |

**Example Output:**
```
Dataset Sizes:
Train samples: 9989 samples
Validation samples: 1109 samples

Batch and epochs: 625
```

#### TensorBoard Setup

```python
        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `datetime.now().strftime('%b%d_%H-%M-%S')` | Current time as string, e.g. `Feb14_15-30-45` |
| 2 | `'SM_MODEL_DIR' in os.environ` | Check if running on SageMaker |
| 3 | Local: `base_dir = 'runs'` | TensorBoard logs saved locally in `./runs/` |
| 4 | SageMaker: `base_dir = '/opt/ml/output/tensorboard'` | SageMaker output path |
| 5 | `SummaryWriter(log_dir=log_dir)` | Create TensorBoard writer for this run |
| 6 | `self.global_step = 0` | Counter for TensorBoard x-axis (incremented per batch) |

**Example:**
```python
# Local training:
log_dir = "runs/run_Feb14_15-30-45"

# SageMaker training:
log_dir = "/opt/ml/output/tensorboard/run_Feb14_15-30-45"

# To view: tensorboard --logdir=runs/
```

#### Optimizer Setup

```python
        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emo_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)
```

| Parameter Group | Learning Rate | Why This Rate |
|----------------|--------------|---------------|
| `text_encoder` (BERT) | 8e-6 = 0.000008 | Very low — BERT is pre-trained on massive data; large updates would destroy learned knowledge |
| `video_encoder` (R3D-18) | 8e-5 = 0.00008 | Low — R3D-18 is pre-trained on Kinetics-400; gentle fine-tuning |
| `audio_encoder` (Conv1d) | 8e-5 = 0.00008 | Low — conv layers are frozen but bias/norm params may update |
| `fusion_layer` | 5e-4 = 0.0005 | Higher — new layer, needs to learn cross-modal interactions from scratch |
| `emo_classifier` | 5e-4 = 0.0005 | Higher — new layer, needs to learn emotion patterns |
| `sentiment_classifier` | 5e-4 = 0.0005 | Higher — new layer, needs to learn sentiment patterns |

**What is `weight_decay=1e-5`?**
- L2 regularization applied to all parameters
- Penalizes large weights: $\text{loss} = \text{loss} + \frac{\lambda}{2} \sum w_i^2$ where $\lambda = 10^{-5}$
- Prevents overfitting by keeping weights small

**What is Adam?**
- Adaptive Moment Estimation optimizer
- Maintains per-parameter learning rates based on 1st moment (mean) and 2nd moment (variance) of gradients
- Generally converges faster than basic SGD

**Example learning rate comparison:**
```
text_encoder:        0.000008  ← barely moves (62.5× slower than fusion)
video_encoder:       0.00008   ← slow          (6.25× slower than fusion)
audio_encoder:       0.00008   ← slow
fusion_layer:        0.0005    ← fastest (learning from scratch)
emo_classifier:      0.0005    ← fastest
sentiment_classifier: 0.0005   ← fastest
```

#### Learning Rate Scheduler

```python
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=2,
        )
```

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `mode='min'` | — | Monitor a value we want to minimize (validation loss) |
| `factor=0.1` | — | Multiply learning rate by 0.1 when reducing (e.g., 5e-4 → 5e-5) |
| `patience=2` | — | Wait 2 epochs without improvement before reducing |

**Example timeline:**
```
Epoch 0: val_loss = 2.31  ← improving
Epoch 1: val_loss = 1.95  ← improving (patience counter = 0)
Epoch 2: val_loss = 1.98  ← worse! (patience counter = 1)
Epoch 3: val_loss = 2.01  ← worse! (patience counter = 2 → REDUCE LR!)
         All learning rates × 0.1:
         fusion_layer: 5e-4 → 5e-5
         text_encoder: 8e-6 → 8e-7
         ...
Epoch 4: val_loss = 1.75  ← improving again with lower LR
```

#### Loss Functions

```python
        self.current_train_losses = None

        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `self.current_train_losses = None` | Placeholder to store last training losses (used in `log_metrics`) |
| 2 | `nn.CrossEntropyLoss(label_smoothing=0.05)` | Loss for emotion (7-class) with label smoothing |
| 3 | `nn.CrossEntropyLoss(label_smoothing=0.05)` | Loss for sentiment (3-class) with label smoothing |

**What CrossEntropyLoss does:**

$$\text{Loss} = -\sum_{i=1}^{C} y_i \cdot \log(\text{softmax}(z_i))$$

Where:
- $C$ = number of classes
- $z$ = raw logits from the model
- $y$ = target label (one-hot)

**What `label_smoothing=0.05` does:**
```python
# Without smoothing (hard labels):
# True class = joy (index 3)
target = [0, 0, 0, 1, 0, 0, 0]
#         anger dig fear JOY neut sad  surp

# With smoothing = 0.05:
# Distribute 0.05 uniformly, keep (1 - 0.05) for true class
target = [0.0071, 0.0071, 0.0071, 0.9571, 0.0071, 0.0071, 0.0071]
#         each gets 0.05/7 = 0.0071    ↑ gets 1 - 0.05 + 0.05/7 = 0.9571

# Why? Prevents the model from becoming overconfident
# Helps generalization by softening the training signal
```

---

### 6b. `log_metrics`

```python
def log_metrics(self, losses, metrics=None, phase='Train'):
    if phase == 'Train':
        self.current_train_losses = losses
    else:
        self.writer.add_scalar('loss/total/train', self.current_train_losses['total'], self.global_step)
        self.writer.add_scalar('loss/total/val', losses['total'], self.global_step)

        self.writer.add_scalar('loss/emotion/train', self.current_train_losses['emotion'], self.global_step)
        self.writer.add_scalar('loss/emotion/val', losses['emotion'], self.global_step)

        self.writer.add_scalar('loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step)
        self.writer.add_scalar('loss/sentiment/val', losses['sentiment'], self.global_step)

    if metrics:
        self.writer.add_scalar(f"{phase}/emotion_precision", metrics['emotion_precision'], self.global_step)
        self.writer.add_scalar(f"{phase}/emotion_accuracy", metrics['emotion_accuracy'], self.global_step)
        self.writer.add_scalar(f"{phase}/sentiment_precision", metrics['sentiment_precision'], self.global_step)
        self.writer.add_scalar(f"{phase}/sentiment_accuracy", metrics['sentiment_accuracy'], self.global_step)
```

| # | Code | When | What It Does |
|---|------|------|-------------|
| 1 | `if phase == 'Train'` | Every training batch | Stores losses temporarily (doesn't write to TensorBoard yet) |
| 2 | `self.current_train_losses = losses` | Training | Saves train losses so they can be paired with val losses later |
| 3 | `else:` (Val/Test) | Every validation/test eval | Writes BOTH train AND val losses to TensorBoard at the same step |
| 4 | `self.writer.add_scalar(tag, value, step)` | Val/Test | Writes a scalar value to TensorBoard: creates a time-series graph |
| 5 | `if metrics:` | When metrics dict is passed | Writes precision and accuracy for emotion/sentiment |

**TensorBoard structure created:**
```
runs/run_Feb14_15-30-45/
├── loss/total/train    ── graph of training total loss
├── loss/total/val      ── graph of validation total loss
├── loss/emotion/train  ── graph of training emotion loss
├── loss/emotion/val    ── graph of validation emotion loss
├── loss/sentiment/train
├── loss/sentiment/val
├── Val/emotion_precision
├── Val/emotion_accuracy
├── Val/sentiment_precision
├── Val/sentiment_accuracy
├── test/emotion_precision  (after final evaluation)
└── test/sentiment_accuracy
```

**Example `add_scalar` call:**
```python
self.writer.add_scalar('loss/total/val', 1.95, 625)
#                       ↑ tag (graph name)  ↑ y-value  ↑ x-value (step)
# Creates point (625, 1.95) on the "loss/total/val" graph
```

---

### 6c. `train_epoch`

```python
def train_epoch(self):
    self.model.train()
    running_loss = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `self.model.train()` | Sets model to training mode: enables `Dropout` and `BatchNorm` uses batch statistics |
| 2 | `running_loss = {...}` | Initialize accumulators for averaging losses at end of epoch |

#### Batch Processing Loop

```python
    for batch in self.train_loader:
        device = next(self.model.parameters()).device

        text_inputs = {
            'input_ids': batch['text_inputs']['input_ids'].to(device),
            'attention_mask': batch['text_inputs']['attention_mask'].to(device)
        }

        video_frames = batch['video_frames'].to(device)
        audio_features = batch['audio_features'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        sentiment_labels = batch['sentiment_labels'].to(device)
```

| # | Code | Explanation |
|---|------|-------------|
| 1 | `for batch in self.train_loader` | Iterate over all batches (e.g., 625 batches × 16 samples) |
| 2 | `next(self.model.parameters()).device` | Gets the device (cuda/cpu) where the model lives — ensures data goes to the same device |
| 3 | `.to(device)` | Moves each tensor from CPU → GPU (or stays on CPU) |

**Why `next(self.model.parameters()).device` instead of a stored variable?**
- Dynamically detects where the model is — works even if model is moved after trainer creation
- `next(...)` gets the first parameter, `.device` checks its location

**Example batch shapes after `.to(device)`:**
```python
text_inputs['input_ids'].shape      # (16, 128) on cuda
text_inputs['attention_mask'].shape  # (16, 128) on cuda
video_frames.shape                   # (16, 30, 3, 224, 224) on cuda
audio_features.shape                 # (16, 1, 64, 300) on cuda
emotion_labels.shape                 # (16,) on cuda  — e.g., tensor([3, 4, 0, 6, ...])
sentiment_labels.shape               # (16,) on cuda  — e.g., tensor([2, 1, 0, 2, ...])
```

#### Forward Pass, Loss, Backward, Update

```python
        # Zero gradients
        self.optimizer.zero_grad()

        # Forward Pass
        outputs = self.model(text_inputs, video_frames, audio_features)

        # Calculate losses using raw logits
        emotion_loss = self.emotion_criterion(outputs['emotion'], emotion_labels)
        sentiment_loss = self.sentiment_criterion(outputs['sentiment'], sentiment_labels)
        total_loss = emotion_loss + sentiment_loss

        # Backward and Calculate gradients
        total_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
```

| # | Code | What It Does |
|---|------|-------------|
| 1 | `self.optimizer.zero_grad()` | Reset all parameter gradients to zero. Without this, gradients from previous batch would accumulate |
| 2 | `self.model(text_inputs, video_frames, audio_features)` | Forward pass through entire model. Returns `{'emotion': (16,7), 'sentiment': (16,3)}` |
| 3 | `self.emotion_criterion(outputs['emotion'], emotion_labels)` | Compute CrossEntropyLoss between predicted logits `(16,7)` and true labels `(16,)`. Returns single scalar |
| 4 | `total_loss = emotion_loss + sentiment_loss` | Joint loss — model is penalized for errors in BOTH tasks |
| 5 | `total_loss.backward()` | **Backpropagation** — computes gradients of loss with respect to every trainable parameter using chain rule |
| 6 | `clip_grad_norm_(..., max_norm=1.0)` | If gradient norm exceeds 1.0, scale all gradients down proportionally. Prevents exploding gradients |
| 7 | `self.optimizer.step()` | Update all parameters: $\theta_{new} = \theta_{old} - lr \cdot \nabla_\theta \text{Loss}$ (simplified; Adam is more complex) |

**The training loop in one diagram:**
```
┌───────────────────────────────────────────┐
│ 1. zero_grad()     — Clear old gradients  │
│                                           │
│ 2. Forward pass    — Input → Predictions  │
│    text ─→ TextEncoder ─→ 128-dim         │
│    video ─→ VideoEncoder ─→ 128-dim       │
│    audio ─→ AudioEncoder ─→ 128-dim       │
│    concat ─→ Fusion ─→ 256-dim            │
│    ├─→ EmotionClassifier ─→ 7 logits      │
│    └─→ SentimentClassifier ─→ 3 logits    │
│                                           │
│ 3. Loss            — Compare to labels    │
│    emotion_loss + sentiment_loss = total   │
│                                           │
│ 4. backward()      — Compute gradients    │
│    ∂Loss/∂w for every trainable weight    │
│                                           │
│ 5. clip_grad_norm_ — Cap gradient size    │
│    Prevents gradient explosion            │
│                                           │
│ 6. step()          — Update weights       │
│    w_new = w - lr * gradient (via Adam)   │
└───────────────────────────────────────────┘
     ↺ Repeat for every batch in epoch
```

**Example of gradient clipping:**
```python
# Before clipping:
# Parameter gradient norms: [0.5, 2.3, 0.1, 4.7]
# Total norm = sqrt(0.5² + 2.3² + 0.1² + 4.7²) = 5.24

# max_norm = 1.0, so scale factor = 1.0 / 5.24 = 0.191

# After clipping:
# [0.5*0.191, 2.3*0.191, 0.1*0.191, 4.7*0.191]
# = [0.095, 0.439, 0.019, 0.897]
# New total norm = 1.0 ✓
```

#### Loss Tracking and Logging

```python
        # Track losses
        running_loss['total'] += total_loss.item()
        running_loss['emotion'] += emotion_loss.item()
        running_loss['sentiment'] += sentiment_loss.item()

        self.log_metrics({
            'total': total_loss.item(),
            'emotion': emotion_loss.item(),
            'sentiment': sentiment_loss.item()
        })

        self.global_step += 1
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `total_loss.item()` | Convert single-element tensor to Python float. Required for accumulation without GPU memory leak |
| 2 | `running_loss['total'] += ...` | Accumulate for epoch average |
| 3 | `self.log_metrics({...})` | Store losses for TensorBoard (phase='Train' by default, so just stores in `current_train_losses`) |
| 4 | `self.global_step += 1` | Increment step counter for TensorBoard x-axis |

**Why `.item()`?**
```python
loss = torch.tensor(1.87, device='cuda', requires_grad=True)

# BAD: running_loss += loss  ← keeps entire computation graph in memory!
# GOOD: running_loss += loss.item()  ← extracts plain float: 1.87
```

#### Return Averaged Losses

```python
    return {
        k: v/len(self.train_loader) for k, v in running_loss.items()
    }
```

**Explanation:** Divides accumulated loss by number of batches to get average loss per batch.

**Example:**
```python
# After 625 batches:
running_loss = {'total': 1168.75, 'emotion': 700.0, 'sentiment': 468.75}

# Return:
{'total': 1168.75/625,     # = 1.87
 'emotion': 700.0/625,     # = 1.12
 'sentiment': 468.75/625}  # = 0.75
```

---

### 6d. `evaluate`

```python
def evaluate(self, data_loader, phase='Val'):
    self.model.eval()
    losses = {'total': 0, 'emotion': 0, 'sentiment': 0}
    all_emotion_preds = []
    all_emotion_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `self.model.eval()` | Sets model to evaluation mode: disables `Dropout`, `BatchNorm` uses running statistics instead of batch statistics |
| 2 | `losses = {...}` | Accumulator for losses |
| 3 | `all_*_preds = []` | Collect ALL predictions across batches for metric computation |
| 4 | `all_*_labels = []` | Collect ALL true labels across batches |

**`.train()` vs `.eval()` behavior:**
```
                     .train()              .eval()
Dropout(0.2):       Drops 20% randomly    No dropout (all neurons active)
BatchNorm:          Uses batch stats      Uses running mean/variance
Gradients:          Computed              Not needed (use inference_mode)
```

#### Inference Loop

```python
    with torch.inference_mode():
        for batch in data_loader:
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }

            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)

            # Forward Pass
            outputs = self.model(text_inputs, video_frames, audio_features)
```

| Key Difference from Training | Explanation |
|------------------------------|-------------|
| `torch.inference_mode()` | Context manager that disables gradient computation and gradient tracking. Faster and uses less memory than `torch.no_grad()` |
| No `optimizer.zero_grad()` | Not needed — no gradients computed |
| No `loss.backward()` | Not needed — no backpropagation |
| No `optimizer.step()` | Not needed — no weight updates |

#### Loss and Prediction Collection

```python
            emotion_loss = self.emotion_criterion(outputs['emotion'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs['sentiment'], sentiment_labels)
            total_loss = emotion_loss + sentiment_loss

            all_emotion_preds.extend(
                outputs["emotion"].argmax(dim=1).cpu().numpy())
            all_emotion_labels.extend(
                emotion_labels.cpu().numpy())

            all_sentiment_preds.extend(
                outputs["sentiment"].argmax(dim=1).cpu().numpy())
            all_sentiment_labels.extend(
                sentiment_labels.cpu().numpy())

            losses['total'] += total_loss.item()
            losses['emotion'] += emotion_loss.item()
            losses['sentiment'] += sentiment_loss.item()
```

| # | Code | Explanation |
|---|------|-------------|
| 1 | `outputs["emotion"].argmax(dim=1)` | Get predicted class index (highest logit) for each sample in the batch |
| 2 | `.cpu()` | Move tensor from GPU → CPU (required for NumPy conversion) |
| 3 | `.numpy()` | Convert PyTorch tensor → NumPy array (required for sklearn metrics) |
| 4 | `.extend(...)` | Append all items from batch to the master list |

**Example of `argmax(dim=1)`:**
```python
# Emotion logits for batch of 4:
outputs['emotion'] = tensor([
    [-0.23, 0.15, -0.87, 1.42, 0.31, -0.56, 0.08],   # → argmax = 3 (joy)
    [0.91, -0.12, 0.03, -0.45, 0.67, 0.12, -0.34],    # → argmax = 0 (anger)
    [-0.11, 0.02, -0.56, 0.23, 1.78, -0.33, 0.05],    # → argmax = 4 (neutral)
    [0.05, 0.12, 0.89, -0.23, -0.45, 0.01, 1.56]      # → argmax = 6 (surprise)
])

outputs['emotion'].argmax(dim=1)  # tensor([3, 0, 4, 6])
```

#### Metric Computation

```python
    avg_loss = {k: v/len(data_loader) for k, v in losses.items()}

    emotion_precision = precision_score(
        all_emotion_labels, all_emotion_preds, average='weighted')
    emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    sentiment_precision = precision_score(
        all_sentiment_labels, all_sentiment_preds, average='weighted')
    sentiment_accuracy = accuracy_score(
        all_sentiment_labels, all_sentiment_preds)
```

| Metric | Function | What It Measures |
|--------|----------|-----------------|
| `precision_score(..., average='weighted')` | sklearn | For each class: TP / (TP + FP), then weighted average by class support (number of true samples per class) |
| `accuracy_score(...)` | sklearn | Overall correct predictions / total predictions |

**Example:**
```python
true_labels = [3, 3, 4, 0, 4, 3, 6]   # joy, joy, neutral, anger, neutral, joy, surprise
predictions = [3, 4, 4, 0, 4, 3, 3]    # joy, neutral, neutral, anger, neutral, joy, joy

accuracy = accuracy_score(true_labels, predictions)
# Correct: positions 0, 2, 3, 4, 5 = 5 out of 7
# accuracy = 5/7 = 0.714

precision = precision_score(true_labels, predictions, average='weighted')
# Per-class precision:
#   anger (0):    1/1 = 1.00  (1 predicted, 1 correct)
#   joy (3):      2/3 = 0.67  (3 predicted, 2 correct)
#   neutral (4):  2/2 = 1.00  (2 predicted, 2 correct)
#   surprise (6): 0/0 = 0.00  (0 predicted)
# Weighted by support (true counts): anger=1, joy=3, neutral=2, surprise=1
# precision = (1*1.00 + 3*0.67 + 2*1.00 + 1*0.00) / 7 = 0.716
```

#### Scheduler Step and Return

```python
    self.log_metrics(avg_loss, {
        'emotion_precision': emotion_precision,
        'emotion_accuracy': emotion_accuracy,
        'sentiment_precision': sentiment_precision,
        'sentiment_accuracy': sentiment_accuracy
    }, phase=phase)

    if phase == 'Val':
        self.scheduler.step(avg_loss['total'])

    return avg_loss, {
        'emotion_precision': emotion_precision,
        'emotion_accuracy': emotion_accuracy,
        'sentiment_precision': sentiment_precision,
        'sentiment_accuracy': sentiment_accuracy
    }
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `self.log_metrics(...)` | Write losses + metrics to TensorBoard |
| 2 | `if phase == 'Val'` | Only adjust learning rate on validation (NOT on test) |
| 3 | `self.scheduler.step(avg_loss['total'])` | Feed validation loss to scheduler — it tracks whether loss is improving and reduces LR after `patience` epochs without improvement |
| 4 | `return avg_loss, {...}` | Return both the loss dict and metrics dict to the caller (`main()` in train.py) |

**Why NOT step scheduler on test?**
- Test data should NEVER influence training decisions
- Scheduler adjusts learning rate (a training decision)
- Test evaluation is purely for final reporting

---

## 7. Main Block (Testing Script)

```python
if __name__ == "__main__":
    dataset = MELDDataset(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits')

    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `if __name__ == "__main__"` | Only runs when executing `python models.py` directly |
| 2 | `MELDDataset(...)` | Load the dataset |
| 3 | `sample = dataset[0]` | Get the first sample (text + video + audio + labels) |
| 4 | `MultimodalSentimentModel()` | Create model with random weights (no training) |
| 5 | `model.eval()` | Set to evaluation mode |

#### Single Sample Inference

```python
    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }

    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)
```

| Code | Before Shape | After Shape | Why `.unsqueeze(0)` |
|------|-------------|-------------|---------------------|
| `input_ids.unsqueeze(0)` | `(128,)` | `(1, 128)` | Model expects batch dimension. `.unsqueeze(0)` adds batch dim of size 1 at position 0 |
| `video_frames.unsqueeze(0)` | `(30, 3, 224, 224)` | `(1, 30, 3, 224, 224)` | Adds batch dimension |
| `audio_features.unsqueeze(0)` | `(1, 64, 300)` | `(1, 1, 64, 300)` | Adds batch dimension |

**Example:**
```python
x = torch.tensor([1, 2, 3])   # shape: (3,)
x.unsqueeze(0)                  # shape: (1, 3)  — added dim at position 0
x.unsqueeze(1)                  # shape: (3, 1)  — added dim at position 1
```

#### Running Inference

```python
    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)

        emotions_probs = torch.softmax(outputs['emotion'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiment'], dim=1)[0]
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `torch.inference_mode()` | No gradient tracking (faster, less memory) |
| 2 | `model(...)` | Forward pass → raw logits |
| 3 | `torch.softmax(outputs['emotion'], dim=1)` | Convert logits to probabilities (sum to 1.0) |
| 4 | `[0]` | Get first (only) sample from batch |

**Softmax formula:**

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

**Example:**
```python
logits = tensor([-0.23, 0.15, -0.87, 1.42, 0.31, -0.56, 0.08])

probs = torch.softmax(logits, dim=0)
# probs = tensor([0.0735, 0.1076, 0.0388, 0.3838, 0.1264, 0.0528, 0.1003])
#                anger   disgust fear   JOY    neutral sadness surprise
#                                       ↑ highest probability

probs.sum()  # tensor(1.0000) — always sums to 1
```

#### Printing Results

```python
    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
        4: 'neutral', 5: 'sadness', 6: 'surprise'
    }

    sentiment_map = {
        0: 'negative', 1: 'neutral', 2: 'positive'
    }

    for i, prob in enumerate(emotions_probs):
        print(f"{emotion_map[i]}: {prob:.2f}")

    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")

    print("Emotion Probabilities:", emotions_probs)
```

**Example Output (with untrained model — nearly uniform random):**
```
anger: 0.14
disgust: 0.15
fear: 0.13
joy: 0.14
neutral: 0.15
sadness: 0.14
surprise: 0.15

negative: 0.33
neutral: 0.34
positive: 0.33

Emotion Probabilities: tensor([0.1381, 0.1515, 0.1289, 0.1412, 0.1519, 0.1389, 0.1495])
```

**After training the probabilities would look like:**
```
anger: 0.02
disgust: 0.01
fear: 0.01
joy: 0.87      ← model is confident
neutral: 0.05
sadness: 0.02
surprise: 0.02
```

---

## 8. Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MultimodalSentimentModel                             │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐      │
│  │  TextEncoder     │  │  VideoEncoder    │  │  AudioEncoder      │      │
│  │                  │  │                  │  │                    │      │
│  │  Input:          │  │  Input:          │  │  Input:            │      │
│  │  input_ids(B,128)│  │  frames          │  │  mel_spec          │      │
│  │  attn_mask(B,128)│  │  (B,30,3,224,224)│  │  (B,1,64,300)      │      │
│  │                  │  │                  │  │                    │      │
│  │  ┌────────────┐  │  │  transpose(1,2)  │  │  squeeze(1)        │      │
│  │  │ BERT       │  │  │  ↓               │  │  ↓                 │      │
│  │  │ (frozen)   │  │  │  (B,3,30,224,224)│  │  (B,64,300)        │      │
│  │  │ 110M params│  │  │                  │  │                    │      │
│  │  └─────┬──────┘  │  │  ┌────────────┐  │  │  ┌──────────────┐  │      │
│  │        ↓          │  │  │ R3D-18     │  │  │  │Conv1d(64,64) │  │      │
│  │  pooler_output    │  │  │ (frozen)   │  │  │  │BatchNorm1d   │  │      │
│  │  (B, 768)         │  │  │ 33M params │  │  │  │ReLU          │  │      │
│  │        ↓          │  │  └─────┬──────┘  │  │  │MaxPool1d(2)  │  │      │
│  │  ┌────────────┐   │  │        ↓          │  │  │Conv1d(64,128)│  │      │
│  │  │Linear      │   │  │  ┌────────────┐   │  │  │BatchNorm1d   │  │      │
│  │  │768 → 128   │   │  │  │Linear      │   │  │  │ReLU          │  │      │
│  │  │(trainable) │   │  │  │512 → 128   │   │  │  │AvgPool1d(1)  │  │      │
│  │  └─────┬──────┘   │  │  │ReLU        │   │  │  └──────┬───────┘  │      │
│  │        ↓          │  │  │Dropout(0.2)│   │  │         ↓          │      │
│  │  Output: (B,128)  │  │  │(trainable) │   │  │  squeeze(-1)       │      │
│  │                   │  │  └─────┬──────┘   │  │  (B, 128)          │      │
│  │                   │  │        ↓          │  │         ↓          │      │
│  │                   │  │  Output: (B,128)  │  │  ┌──────────────┐  │      │
│  │                   │  │                   │  │  │Linear 128→128│  │      │
│  │                   │  │                   │  │  │ReLU          │  │      │
│  │                   │  │                   │  │  │Dropout(0.2)  │  │      │
│  │                   │  │                   │  │  └──────┬───────┘  │      │
│  │                   │  │                   │  │         ↓          │      │
│  │                   │  │                   │  │  Output: (B,128)   │      │
│  └────────┬──────────┘  └────────┬──────────┘  └─────────┬──────────┘      │
│           │                      │                       │                 │
│           └──────────────────────┼───────────────────────┘                 │
│                                  ↓                                         │
│                         torch.cat(dim=1)                                   │
│                          (B, 384)                                          │
│                                  ↓                                         │
│                  ┌───────────────────────────────┐                         │
│                  │       Fusion Layer             │                         │
│                  │  Linear(384 → 256)             │                         │
│                  │  BatchNorm1d(256)               │                         │
│                  │  ReLU                           │                         │
│                  │  Dropout(0.3)                   │                         │
│                  └───────────────┬───────────────┘                         │
│                                 ↓                                          │
│                          (B, 256)                                          │
│                        ┌────────┴────────┐                                 │
│                        ↓                 ↓                                 │
│          ┌──────────────────┐  ┌──────────────────┐                        │
│          │ Emotion Classif. │  │ Sentiment Classif.│                        │
│          │ Linear(256→128)  │  │ Linear(256→128)   │                        │
│          │ ReLU             │  │ ReLU              │                        │
│          │ Dropout(0.2)     │  │ Dropout(0.2)      │                        │
│          │ Linear(128→7)    │  │ Linear(128→3)     │                        │
│          └────────┬─────────┘  └────────┬─────────┘                        │
│                   ↓                     ↓                                  │
│            (B, 7) logits         (B, 3) logits                             │
│                                                                            │
│            anger    0             negative  0                               │
│            disgust  1             neutral   1                               │
│            fear     2             positive  2                               │
│            joy      3                                                      │
│            neutral  4                                                      │
│            sadness  5                                                      │
│            surprise 6                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Tensor Shape Cheat Sheet

Complete tensor shapes flowing through the model (batch_size=16):

| Stage | Tensor | Shape |
|-------|--------|-------|
| **Input** | `input_ids` | `(16, 128)` |
| | `attention_mask` | `(16, 128)` |
| | `video_frames` | `(16, 30, 3, 224, 224)` |
| | `audio_features` | `(16, 1, 64, 300)` |
| | `emotion_labels` | `(16,)` |
| | `sentiment_labels` | `(16,)` |
| **TextEncoder** | BERT output | `(16, 768)` |
| | After projection | `(16, 128)` |
| **VideoEncoder** | After transpose | `(16, 3, 30, 224, 224)` |
| | After R3D-18 + FC | `(16, 128)` |
| **AudioEncoder** | After squeeze | `(16, 64, 300)` |
| | After Conv1d block | `(16, 128, 1)` |
| | After squeeze(-1) | `(16, 128)` |
| | After projection | `(16, 128)` |
| **Fusion** | After cat | `(16, 384)` |
| | After fusion_layer | `(16, 256)` |
| **Classifiers** | Emotion output | `(16, 7)` |
| | Sentiment output | `(16, 3)` |
| **Loss** | emotion_loss | scalar |
| | sentiment_loss | scalar |
| | total_loss | scalar |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Logits** | Raw, unnormalized model outputs before softmax |
| **Softmax** | Converts logits to probabilities that sum to 1 |
| **CrossEntropyLoss** | Measures difference between predicted probabilities and true labels |
| **Label Smoothing** | Softens one-hot targets to prevent overconfidence |
| **Backpropagation** | Algorithm to compute gradients via chain rule |
| **Gradient Clipping** | Limits gradient magnitude to prevent exploding gradients |
| **Frozen Parameters** | `requires_grad=False` — no gradient computation, no weight updates |
| **Batch Normalization** | Normalizes layer inputs across the batch for stable training |
| **Dropout** | Randomly zeroes neurons during training to prevent overfitting |
| **AdaptiveAvgPool** | Averages a dimension to a fixed output size regardless of input size |
| **Mel Spectrogram** | Frequency representation of audio using perceptually-spaced mel bands |
| **[CLS] Token** | Special BERT token whose embedding represents the entire input sentence |
| **R3D-18** | 3D ResNet-18 — convolves over spatial AND temporal dimensions for video |
| **ReduceLROnPlateau** | Reduces learning rate when a monitored metric stops improving |
