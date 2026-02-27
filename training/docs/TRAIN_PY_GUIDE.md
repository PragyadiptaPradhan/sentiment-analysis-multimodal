# train.py — Complete Line-by-Line Documentation with Examples

> This guide walks through every single line and function in `train.py`, explaining what it does, why it's needed, and providing concrete examples and expected outputs.

---

## Table of Contents

- [1. Imports](#1-imports)
- [2. SageMaker Environment Variables](#2-sagemaker-environment-variables)
- [3. CUDA Memory Configuration](#3-cuda-memory-configuration)
- [4. parse_args() Function](#4-parse_args-function)
- [5. main() Function](#5-main-function)
  - [5a. FFmpeg Check](#5a-ffmpeg-check)
  - [5b. Audio Backend Listing](#5b-audio-backend-listing)
  - [5c. Argument Parsing & Device Selection](#5c-argument-parsing--device-selection)
  - [5d. Initial GPU Memory Tracking](#5d-initial-gpu-memory-tracking)
  - [5e. DataLoader Preparation](#5e-dataloader-preparation)
  - [5f. Debug Path Printing](#5f-debug-path-printing)
  - [5g. Model & Trainer Initialization](#5g-model--trainer-initialization)
  - [5h. Metrics Dictionary Setup](#5h-metrics-dictionary-setup)
  - [5i. Training Loop](#5i-training-loop)
  - [5j. Metrics Tracking Inside Loop](#5j-metrics-tracking-inside-loop)
  - [5k. SageMaker JSON Logging](#5k-sagemaker-json-logging)
  - [5l. GPU Memory Monitoring Per Epoch](#5l-gpu-memory-monitoring-per-epoch)
  - [5m. Best Model Checkpointing](#5m-best-model-checkpointing)
  - [5n. Test Set Evaluation](#5n-test-set-evaluation)
  - [5o. Final Test Metrics Logging](#5o-final-test-metrics-logging)
- [6. Entry Point Guard](#6-entry-point-guard)
- [7. End-to-End Execution Walkthrough](#7-end-to-end-execution-walkthrough)
- [8. Relationship with Other Modules](#8-relationship-with-other-modules)
- [9. Frequently Asked Questions](#9-frequently-asked-questions)

---

## 1. Imports

```python
import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys
```

### Standard Library Imports

| # | Import | What It Does | Example in This File |
|---|--------|-------------|----------------------|
| 1 | `import os` | File/directory operations and environment variables | `os.environ.get('SM_MODEL_DIR', '.')` reads env vars; `os.path.join(dir, 'file.csv')` builds paths |
| 2 | `import argparse` | Parses command-line arguments into Python objects | `parser.add_argument("--epochs", type=int, default=20)` |
| 3 | `import json` | Serializes Python dicts to JSON strings | `json.dumps({"metrics": [...]})` outputs SageMaker-compatible logs |
| 4 | `import sys` | System-level operations (exit, stdin/stdout) | `sys.exit(1)` halts the program with error code 1 |

### Third-Party Imports

| # | Import | Library | What It Does | Example in This File |
|---|--------|---------|-------------|----------------------|
| 5 | `import torchaudio` | PyTorch Audio | Audio processing utilities | `torchaudio.list_audio_backends()` lists installed backends |
| 6 | `import torch` | PyTorch | Core deep learning framework | `torch.device('cuda')`, `torch.save(...)`, `torch.cuda.is_available()` |
| 7 | `from tqdm import tqdm` | tqdm | Progress bar wrapper for loops | `for epoch in tqdm(range(20), desc="Epochs"):` |

### Local Module Imports

```python
from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg
```

| Import | Source File | What It Provides |
|--------|------------|-----------------|
| `prepare_dataloaders` | `meld_dataset.py` | Function that creates 3 PyTorch DataLoaders (train/val/test) from CSV files and video directories |
| `MultimodalSentimentModel` | `models.py` | Neural network with TextEncoder (BERT) + VideoEncoder (R3D-18) + AudioEncoder (Conv1d) + Fusion + 2 Classifiers |
| `MultimodalTrainer` | `models.py` | Training loop manager with optimizer, scheduler, loss functions, TensorBoard logging, and evaluation |
| `install_ffmpeg` | `install_ffmpeg.py` | Ensures FFmpeg is available for audio extraction from video files |

**How these relate:**
```
train.py (orchestrator)
├── install_ffmpeg.py  → ensures FFmpeg is installed
├── meld_dataset.py    → loads CSV + videos → creates DataLoaders
└── models.py          → defines model architecture + training logic
```

---

## 2. SageMaker Environment Variables

```python
# AWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get(
    'SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get(
    'SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get(
    'SM_CHANNEL_TEST', "/opt/ml/input/data/test")
```

### Line-by-Line

| Variable | Environment Key | Default (Local) | SageMaker Value | Purpose |
|----------|-----------------|------------------|-----------------|---------|
| `SM_MODEL_DIR` | `SM_MODEL_DIR` | `"."` (current dir) | `/opt/ml/model` | Where to save trained model artifacts |
| `SM_CHANNEL_TRAINING` | `SM_CHANNEL_TRAINING` | `/opt/ml/input/data/training` | `/opt/ml/input/data/training` | Path to training dataset |
| `SM_CHANNEL_VALIDATION` | `SM_CHANNEL_VALIDATION` | `/opt/ml/input/data/validation` | `/opt/ml/input/data/validation` | Path to validation dataset |
| `SM_CHANNEL_TEST` | `SM_CHANNEL_TEST` | `/opt/ml/input/data/test` | `/opt/ml/input/data/test` | Path to test dataset |

### How `os.environ.get()` Works

```python
os.environ.get('SM_MODEL_DIR', ".")
#              ↑ key to look up   ↑ fallback if key doesn't exist
```

**Example — Running Locally:**
```python
# SM_MODEL_DIR is NOT set in environment
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
print(SM_MODEL_DIR)  # → "."
# Model will be saved as ./model.pth
```

**Example — Running on SageMaker:**
```python
# SageMaker automatically sets SM_MODEL_DIR=/opt/ml/model
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
print(SM_MODEL_DIR)  # → "/opt/ml/model"
# Model will be saved as /opt/ml/model/model.pth
```

---

## 3. CUDA Memory Configuration

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
```

| Detail | Explanation |
|--------|-------------|
| **What** | Sets PyTorch's CUDA memory allocator to use expandable segments |
| **Why** | Prevents `CUDA out of memory` errors by allowing dynamic memory growth |
| **When matters** | Large batch sizes, long video sequences, or limited GPU VRAM |
| **Default behavior** | PyTorch pre-allocates fixed memory blocks; if a block doesn't fit, it OOMs |
| **With this setting** | Memory segments can grow dynamically, reducing fragmentation |

**Example — Without this setting:**
```
RuntimeError: CUDA out of memory. Tried to allocate 512 MiB
(GPU 0; 8.00 GiB total capacity; 7.23 GiB already allocated)
```

**Example — With this setting:**
```
GPU dynamically expands segments → training continues normally
```

---

## 4. `parse_args()` Function

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # Data directories
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()
```

### Line-by-Line

| # | Code | What It Does |
|---|------|-------------|
| 1 | `parser = argparse.ArgumentParser()` | Creates a new argument parser object |
| 2 | `parser.add_argument("--epochs", type=int, default=20)` | Register `--epochs` flag: expects integer, defaults to 20 |
| 3 | `parser.add_argument("--batch-size", type=int, default=16)` | Register `--batch-size`: expects integer, defaults to 16. Accessed as `args.batch_size` (hyphen → underscore) |
| 4 | `parser.add_argument("--learning-rate", type=float, default=0.001)` | Register `--learning-rate`: expects float, defaults to 0.001 |
| 5 | `parser.add_argument("--train-dir", ...)` | Training data directory; defaults to `SM_CHANNEL_TRAINING` |
| 6 | `parser.add_argument("--val-dir", ...)` | Validation data directory; defaults to `SM_CHANNEL_VALIDATION` |
| 7 | `parser.add_argument("--test-dir", ...)` | Test data directory; defaults to `SM_CHANNEL_TEST` |
| 8 | `parser.add_argument("--model-dir", ...)` | Model save directory; defaults to `SM_MODEL_DIR` |
| 9 | `return parser.parse_args()` | Parse `sys.argv`, return `Namespace` object with all arguments |

### Usage Examples

```bash
# Example 1: All defaults
python train.py
# args.epochs = 20
# args.batch_size = 16
# args.learning_rate = 0.001
# args.train_dir = "/opt/ml/input/data/training"

# Example 2: Override training params
python train.py --epochs 50 --batch-size 8 --learning-rate 0.0005
# args.epochs = 50
# args.batch_size = 8
# args.learning_rate = 0.0005

# Example 3: Override data paths (local training)
python train.py --train-dir ./dataset/train --val-dir ./dataset/dev --test-dir ./dataset/test --model-dir ./saved_models
# args.train_dir = "./dataset/train"
# args.model_dir = "./saved_models"

# Example 4: SageMaker invokes the script internally like:
python train.py --epochs 20 --batch-size 16 --model-dir /opt/ml/model --train-dir /opt/ml/input/data/training
```

### Return Value

```python
args = parse_args()
# args is a Namespace:
# Namespace(epochs=20, batch_size=16, learning_rate=0.001,
#           train_dir='/opt/ml/input/data/training',
#           val_dir='/opt/ml/input/data/validation',
#           test_dir='/opt/ml/input/data/test',
#           model_dir='.')
```

---

## 5. `main()` Function

The core function that orchestrates the entire training pipeline.

---

### 5a. FFmpeg Check

```python
if not install_ffmpeg():
    print("Error: FFmpeg installation failed. Cannot continue training.")
    sys.exit(1)
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `install_ffmpeg()` | Calls helper function that checks if FFmpeg is installed, installs it if not, returns `True`/`False` |
| 2 | `if not install_ffmpeg()` | If FFmpeg is NOT available after the check |
| 3 | `print("Error: ...")` | Print error message to stdout |
| 4 | `sys.exit(1)` | Terminate the program with exit code 1 (error). Code 0 = success, 1 = failure |

**Why FFmpeg is needed:**
- `meld_dataset.py` uses FFmpeg to extract audio from `.mp4` video files
- Without FFmpeg, audio features cannot be computed
- Training cannot proceed without all three modalities (text, video, audio)

**Example:**
```
# FFmpeg installed:
install_ffmpeg() → True → continue

# FFmpeg NOT installed and install fails:
install_ffmpeg() → False
→ "Error: FFmpeg installation failed. Cannot continue training."
→ Program exits with code 1
```

---

### 5b. Audio Backend Listing

```python
print("Available audio backends:")
print(str(torchaudio.list_audio_backends()))
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `"Available audio backends:"` | Header print for readability |
| 2 | `torchaudio.list_audio_backends()` | Returns list of available audio I/O backends |

**Example Outputs (varies by system):**
```
# Linux with sox and soundfile:
Available audio backends:
['sox', 'soundfile']

# Windows with soundfile only:
Available audio backends:
['soundfile']

# Minimal install:
Available audio backends:
[]
```

**Purpose:** Debugging — if audio loading fails later, this log helps identify missing backends.

---

### 5c. Argument Parsing & Device Selection

```python
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

| Line | Code | What Happens |
|------|------|-------------|
| 1 | `args = parse_args()` | Calls `parse_args()` (see [Section 4](#4-parse_args-function)), stores result |
| 2 | `torch.cuda.is_available()` | Returns `True` if NVIDIA GPU + CUDA drivers are available |
| 3 | `torch.device('cuda')` | Creates a device object representing the GPU |
| 4 | `torch.device('cpu')` | Creates a device object representing the CPU |

**Example:**
```python
# Machine with NVIDIA GPU:
torch.cuda.is_available()  # True
device = torch.device('cuda')
print(device)  # cuda

# Machine without GPU:
torch.cuda.is_available()  # False
device = torch.device('cpu')
print(device)  # cpu

# Later usage:
model.to(device)        # Moves model weights to GPU/CPU
tensor.to(device)       # Moves data tensors to GPU/CPU
```

---

### 5d. Initial GPU Memory Tracking

```python
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    memory_used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Initial GPU memory used: {memory_used:.2f} GB")
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `if torch.cuda.is_available()` | Only run memory tracking if GPU exists |
| 2 | `torch.cuda.reset_peak_memory_stats()` | Reset the "high water mark" memory counter to 0 |
| 3 | `torch.cuda.max_memory_allocated()` | Returns peak GPU memory used (in bytes) |
| 4 | `/ 1024**3` | Convert bytes → gigabytes: $\frac{\text{bytes}}{1024^3}$ |
| 5 | `f"...{memory_used:.2f} GB"` | Format to 2 decimal places |

**Example:**
```
# Right after reset, before model loading:
Initial GPU memory used: 0.00 GB

# If some CUDA context already exists:
Initial GPU memory used: 0.12 GB
```

**Why reset first?** If any CUDA operation happened before (library initialization), resetting gives a clean baseline.

---

### 5e. DataLoader Preparation

```python
train_loader, val_loader, test_loader = prepare_dataloaders(
    train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
    train_video_dir=os.path.join(args.train_dir, 'train_splits'),
    dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
    dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
    test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
    test_video_dir=os.path.join(
        args.test_dir, 'output_repeated_splits_test'),
    batch_size=args.batch_size
)
```

| Parameter | Built Path (using defaults) | File/Dir Contents |
|-----------|---------------------------|-------------------|
| `train_csv` | `/opt/ml/input/data/training/train_sent_emo.csv` | CSV with columns: Utterance, Emotion, Sentiment, Dialogue_ID, Utterance_ID |
| `train_video_dir` | `/opt/ml/input/data/training/train_splits` | `.mp4` files named `dia{X}_utt{Y}.mp4` |
| `dev_csv` | `/opt/ml/input/data/validation/dev_sent_emo.csv` | Validation CSV data |
| `dev_video_dir` | `/opt/ml/input/data/validation/dev_splits_complete` | Validation video clips |
| `test_csv` | `/opt/ml/input/data/test/test_sent_emo.csv` | Test CSV data |
| `test_video_dir` | `/opt/ml/input/data/test/output_repeated_splits_test` | Test video clips |
| `batch_size` | `16` (default) | Number of samples per batch |

### How `os.path.join()` Works

```python
os.path.join('/opt/ml/input/data/training', 'train_sent_emo.csv')
# Linux: '/opt/ml/input/data/training/train_sent_emo.csv'
# Windows: '\\opt\\ml\\input\\data\\training\\train_sent_emo.csv'
```

### Return Values

```python
# train_loader: DataLoader wrapping ~9989 samples, yields batches of 16
# val_loader:   DataLoader wrapping ~1109 samples, yields batches of 16
# test_loader:  DataLoader wrapping ~2610 samples, yields batches of 16

# Each batch is a dictionary:
# {
#   'text_inputs': {'input_ids': (16, 128), 'attention_mask': (16, 128)},
#   'video_frames': (16, 30, 3, 224, 224),
#   'audio_features': (16, 1, 64, 300),
#   'emotion_labels': (16,),
#   'sentiment_labels': (16,)
# }
```

**Underneath, `prepare_dataloaders` does:**
1. Creates 3 `MELDDataset` objects (one per split)
2. Wraps each in a `DataLoader` with `collate_fn` that filters out `None` samples
3. Train loader has `shuffle=True`, val/test have `shuffle=False`

---

### 5f. Debug Path Printing

```python
print(f"""Training DSV path: {os.path.join(
    args.train_dir, 'train_sent_emo.csv')}""")
print(f"""Training video directory: {
      os.path.join(args.train_dir, 'train_splits')}""")
```

| Line | Purpose |
|------|---------|
| 1 | Print full path to training CSV (for debugging path issues) |
| 2 | Print full path to training video directory |

**Example Output:**
```
Training DSV path: /opt/ml/input/data/training/train_sent_emo.csv
Training video directory: /opt/ml/input/data/training/train_splits
```

**Why:** On SageMaker, data is mounted into containers. Printing paths helps verify that data channels are mapped correctly.

---

### 5g. Model & Trainer Initialization

```python
model = MultimodalSentimentModel().to(device)
trainer = MultimodalTrainer(model, train_loader, val_loader)
best_val_loss = float('inf')
```

| # | Code | What It Does |
|---|------|-------------|
| 1 | `MultimodalSentimentModel()` | Creates the neural network with: TextEncoder (BERT, frozen, projects 768→128), VideoEncoder (R3D-18, frozen, projects→128), AudioEncoder (Conv1d, frozen, projects→128), FusionLayer (384→256), EmotionClassifier (256→7), SentimentClassifier (256→3) |
| 2 | `.to(device)` | Moves all model parameters to GPU or CPU |
| 3 | `MultimodalTrainer(model, train_loader, val_loader)` | Creates trainer object that sets up: Adam optimizer with per-layer learning rates, ReduceLROnPlateau scheduler, CrossEntropyLoss with label smoothing, TensorBoard writer |
| 4 | `best_val_loss = float('inf')` | Initialize to positive infinity so any real loss will be "better" |

**Example of `float('inf')`:**
```python
best_val_loss = float('inf')   # ∞
print(0.85 < best_val_loss)    # True — any number is less than infinity
print(float('inf') < best_val_loss)  # False — infinity is NOT less than infinity
```

### What Happens Inside `MultimodalTrainer.__init__()`:
```python
# Prints dataset info:
# Dataset Sizes:
# Train samples: 9989 samples
# Validation samples: 1109 samples
# Batch and epochs: 625

# Sets up Adam optimizer with different learning rates:
# text_encoder:          lr = 8e-6   (very low — BERT is pre-trained)
# video_encoder:         lr = 8e-5   (low — R3D-18 is pre-trained)
# audio_encoder:         lr = 8e-5   (low — mostly frozen)
# fusion_layer:          lr = 5e-4   (higher — new layer, needs to learn)
# emo_classifier:        lr = 5e-4   (higher — new layer)
# sentiment_classifier:  lr = 5e-4   (higher — new layer)

# Sets up ReduceLROnPlateau:
# Reduces lr by 10x if val loss doesn't improve for 2 epochs

# Sets up CrossEntropyLoss with label_smoothing=0.05:
# Instead of hard targets [0, 0, 1, 0] → soft targets [0.007, 0.007, 0.957, 0.007]
```

---

### 5h. Metrics Dictionary Setup

```python
metrics_data = {
    "train_losses": [],
    "val_losses": [],
    "epochs": []
}
```

| Key | Type | Purpose |
|-----|------|---------|
| `"train_losses"` | `list[float]` | Stores average training loss per epoch |
| `"val_losses"` | `list[float]` | Stores average validation loss per epoch |
| `"epochs"` | `list[int]` | Stores epoch numbers (0, 1, 2, ...) |

**Example after 3 epochs:**
```python
metrics_data = {
    "train_losses": [2.14, 1.87, 1.63],
    "val_losses":   [2.31, 1.95, 1.78],
    "epochs":       [0, 1, 2]
}
```

---

### 5i. Training Loop

```python
for epoch in tqdm(range(args.epochs), desc="Epochs"):
    train_loss = trainer.train_epoch()
    val_loss, val_metrics = trainer.evaluate(val_loader)
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `range(args.epochs)` | Creates sequence `[0, 1, 2, ..., 19]` for 20 epochs |
| 2 | `tqdm(..., desc="Epochs")` | Wraps the range with a progress bar labeled "Epochs" |
| 3 | `trainer.train_epoch()` | Runs one full pass through training data, returns average losses dict |
| 4 | `trainer.evaluate(val_loader)` | Runs inference on validation data, returns (losses_dict, metrics_dict) |

**tqdm Output Example:**
```
Epochs:  15%|█▌        | 3/20 [05:23<30:28, 107.56s/it]
         ↑             ↑     ↑       ↑        ↑
      label      completed  elapsed  remaining  speed
```

### What `trainer.train_epoch()` Returns:
```python
train_loss = {
    "total": 1.87,        # emotion_loss + sentiment_loss averaged
    "emotion": 1.12,      # emotion CrossEntropyLoss averaged
    "sentiment": 0.75     # sentiment CrossEntropyLoss averaged
}
```

### What `trainer.evaluate()` Returns:
```python
val_loss = {
    "total": 1.95,
    "emotion": 1.18,
    "sentiment": 0.77
}

val_metrics = {
    "emotion_precision": 0.42,    # Weighted precision across 7 emotions
    "emotion_accuracy": 0.38,     # % of correctly predicted emotions
    "sentiment_precision": 0.61,  # Weighted precision across 3 sentiments
    "sentiment_accuracy": 0.58    # % of correctly predicted sentiments
}
```

### Inside `trainer.train_epoch()` — Step by Step:
```
For each batch (16 samples):
  1. Move data to GPU: text_inputs, video_frames, audio_features, labels
  2. Zero gradients: optimizer.zero_grad()
  3. Forward pass: outputs = model(text, video, audio)
     → outputs = {'emotion': (16, 7), 'sentiment': (16, 3)}  ← raw logits
  4. Compute loss: emotion_loss + sentiment_loss = total_loss
  5. Backward pass: total_loss.backward()  ← compute gradients
  6. Clip gradients: clip_grad_norm_(max_norm=1.0)  ← prevent exploding gradients
  7. Update weights: optimizer.step()
  8. Log to TensorBoard
```

---

### 5j. Metrics Tracking Inside Loop

```python
# Track metrics
metrics_data["train_losses"].append(train_loss["total"])
metrics_data["val_losses"].append(val_loss["total"])
metrics_data["epochs"].append(epoch)
```

| Line | Code | What It Does |
|------|------|-------------|
| 1 | `.append(train_loss["total"])` | Adds this epoch's average training loss to the history list |
| 2 | `.append(val_loss["total"])` | Adds this epoch's average validation loss to the history list |
| 3 | `.append(epoch)` | Adds the epoch number (0-indexed) |

**Example — Epoch 5:**
```python
# Before:
metrics_data["train_losses"] = [2.14, 1.87, 1.63, 1.45, 1.32]
# After:
metrics_data["train_losses"] = [2.14, 1.87, 1.63, 1.45, 1.32, 1.21]
#                                                               ↑ new
```

---

### 5k. SageMaker JSON Logging

```python
print(json.dumps({
    "metrics": [
        {"Name": "train:loss", "Value": train_loss["total"]},
        {"Name": "validation:loss", "Value": val_loss["total"]},
        {"Name": "validation:emotion_precision",
            "Value": val_metrics["emotion_precision"]},
        {"Name": "validation:emotion_accuracy",
            "Value": val_metrics["emotion_accuracy"]},
        {"Name": "validation:sentiment_precision",
            "Value": val_metrics["sentiment_precision"]},
        {"Name": "validation:sentiment_accuracy",
            "Value": val_metrics["sentiment_accuracy"]},
    ]
}))
```

| Component | Explanation |
|-----------|-------------|
| `json.dumps({...})` | Converts dict → JSON string |
| `"Name": "train:loss"` | SageMaker metric name (appears in CloudWatch) |
| `"Value": train_loss["total"]` | Actual numeric metric value |
| Format: `{"metrics": [...]}` | **SageMaker-specific format** — SageMaker parses this from stdout |

**Example Output (one line of JSON):**
```json
{"metrics": [{"Name": "train:loss", "Value": 1.87}, {"Name": "validation:loss", "Value": 1.95}, {"Name": "validation:emotion_precision", "Value": 0.42}, {"Name": "validation:emotion_accuracy", "Value": 0.38}, {"Name": "validation:sentiment_precision", "Value": 0.61}, {"Name": "validation:sentiment_accuracy", "Value": 0.58}]}
```

**What SageMaker does with this:**
```
stdout → SageMaker log parser → CloudWatch Metrics
                                   ├── train:loss = 1.87
                                   ├── validation:loss = 1.95
                                   ├── validation:emotion_precision = 0.42
                                   ├── validation:emotion_accuracy = 0.38
                                   ├── validation:sentiment_precision = 0.61
                                   └── validation:sentiment_accuracy = 0.58
```

You can then view these as real-time graphs in the SageMaker Training Job console.

---

### 5l. GPU Memory Monitoring Per Epoch

```python
if torch.cuda.is_available():
    memory_used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory used: {memory_used:.2f} GB")
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `if torch.cuda.is_available()` | Skip if no GPU |
| 2 | `torch.cuda.max_memory_allocated()` | Returns peak bytes allocated since last reset |
| 3 | `/ 1024**3` | Convert bytes to GB: $1 \text{ GB} = 1024^3 \text{ bytes} = 1{,}073{,}741{,}824 \text{ bytes}$ |
| 4 | `f"...{memory_used:.2f} GB"` | Print with 2 decimal places |

**Example across epochs:**
```
Epoch 0: Peak GPU memory used: 3.42 GB
Epoch 1: Peak GPU memory used: 3.42 GB  ← stable = good
Epoch 2: Peak GPU memory used: 3.43 GB
...
Epoch 15: Peak GPU memory used: 5.61 GB  ← growing = possible memory leak!
```

**Why monitor?**
- Detect memory leaks (gradual increase)
- Right-size GPU instance (if peak is 4GB, don't pay for 24GB GPU)
- Debug OOM errors

---

### 5m. Best Model Checkpointing

```python
# Save best model
if val_loss["total"] < best_val_loss:
    best_val_loss = val_loss["total"]
    torch.save(model.state_dict(), os.path.join(
        args.model_dir, "model.pth"))
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `if val_loss["total"] < best_val_loss` | Is current validation loss the best we've seen? |
| 2 | `best_val_loss = val_loss["total"]` | Update the best recorded loss |
| 3 | `model.state_dict()` | Get all model weights as an OrderedDict |
| 4 | `torch.save(..., os.path.join(args.model_dir, "model.pth"))` | Save weights to disk |

### How Checkpointing Progresses:

```
Epoch 0: val_loss = 2.31 → 2.31 < inf ✓  → SAVE model.pth, best = 2.31
Epoch 1: val_loss = 1.95 → 1.95 < 2.31 ✓ → SAVE model.pth, best = 1.95
Epoch 2: val_loss = 1.78 → 1.78 < 1.95 ✓ → SAVE model.pth, best = 1.78
Epoch 3: val_loss = 1.82 → 1.82 < 1.78 ✗ → SKIP (model overfitting)
Epoch 4: val_loss = 1.71 → 1.71 < 1.78 ✓ → SAVE model.pth, best = 1.71
Epoch 5: val_loss = 1.75 → 1.75 < 1.71 ✗ → SKIP
...
```

**Key insight:** Only the best-performing model is saved. This prevents saving an overfitted model from a later epoch.

### What `model.state_dict()` Contains:

```python
state_dict = model.state_dict()
# OrderedDict with entries like:
# {
#   'text_encoder.bert.embeddings.word_embeddings.weight': tensor(30522, 768),
#   'text_encoder.projection.weight': tensor(128, 768),
#   'video_encoder.backbone.layer1.0.conv1.weight': tensor(64, 64, 3, 3, 3),
#   'fusion_layer.0.weight': tensor(256, 384),
#   'emo_classifier.3.weight': tensor(7, 128),
#   'sentiment_classifier.3.weight': tensor(3, 128),
#   ...
# }
```

### Loading the Saved Model Later:
```python
model = MultimodalSentimentModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

---

### 5n. Test Set Evaluation

```python
# After training is complete, evaluate on test set
print("Evaluating on test set...")
test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
metrics_data["test_loss"] = test_loss["total"]
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `print("Evaluating on test set...")` | Status message (training loop is done) |
| 2 | `trainer.evaluate(test_loader, phase="test")` | Run model on test data; `phase="test"` controls TensorBoard logging prefix |
| 3 | `metrics_data["test_loss"] = test_loss["total"]` | Store final test loss |

**Key Difference: `phase="test"` vs `phase="Val"`**
- When `phase="Val"`: `self.scheduler.step(avg_loss['total'])` is called (adjusts learning rate)
- When `phase="test"`: Scheduler is NOT called (test set should not influence training)

**Example:**
```python
test_loss = {"total": 1.82, "emotion": 1.10, "sentiment": 0.72}
test_metrics = {
    "emotion_accuracy": 0.41,
    "sentiment_accuracy": 0.59,
    "emotion_precision": 0.39,
    "sentiment_precision": 0.57
}
```

**Important:** This evaluates the LAST epoch's model, not the best checkpoint. The best model was already saved to `model.pth`.

---

### 5o. Final Test Metrics Logging

```python
print(json.dumps({
    "metrics": [
        {"Name": "test:loss", "Value": test_loss["total"]},
        {"Name": "test:emotion_accuracy",
            "Value": test_metrics["emotion_accuracy"]},
        {"Name": "test:sentiment_accuracy",
            "Value": test_metrics["sentiment_accuracy"]},
        {"Name": "test:emotion_precision",
            "Value": test_metrics["emotion_precision"]},
        {"Name": "test:sentiment_precision",
            "Value": test_metrics["sentiment_precision"]},
    ]
}))
```

Same SageMaker JSON format as [Section 5k](#5k-sagemaker-json-logging), but with test-set metrics.

**Example Output:**
```json
{"metrics": [{"Name": "test:loss", "Value": 1.82}, {"Name": "test:emotion_accuracy", "Value": 0.41}, {"Name": "test:sentiment_accuracy", "Value": 0.59}, {"Name": "test:emotion_precision", "Value": 0.39}, {"Name": "test:sentiment_precision", "Value": 0.57}]}
```

---

## 6. Entry Point Guard

```python
if __name__ == "__main__":
    main()
```

| Code | Explanation |
|------|-------------|
| `__name__` | Built-in Python variable. Equals `"__main__"` when the file is run directly |
| `if __name__ == "__main__"` | Only execute `main()` when running `python train.py` directly |
| Why needed | If another file does `from train import parse_args`, `main()` would NOT run (preventing accidental training) |

**Example:**
```python
# Running directly:
python train.py
# __name__ == "__main__" → True → main() executes

# Importing in another file:
from train import parse_args
# __name__ == "train" → False → main() does NOT execute
```

---

## 7. End-to-End Execution Walkthrough

Here is the complete sequence of events when you run `python train.py --epochs 3 --batch-size 8`:

```
Step 1: install_ffmpeg()
        → FFmpeg check/install → True ✓

Step 2: Print audio backends
        → "Available audio backends: ['soundfile']"

Step 3: parse_args()
        → args.epochs = 3, args.batch_size = 8, ...

Step 4: Device selection
        → device = cuda (or cpu)

Step 5: GPU memory baseline
        → "Initial GPU memory used: 0.00 GB"

Step 6: prepare_dataloaders(...)
        → Loads CSVs, creates MELDDataset objects
        → Wraps in DataLoaders: train(1249 batches), val(139 batches), test(327 batches)

Step 7: Print debug paths
        → "Training DSV path: ..."
        → "Training video directory: ..."

Step 8: Create model
        → MultimodalSentimentModel on GPU
        → ~118M parameters (mostly frozen BERT + R3D-18)

Step 9: Create trainer
        → Prints "Dataset Sizes: Train: 9989, Val: 1109"
        → Sets up Adam optimizer, scheduler, loss functions, TensorBoard

Step 10: best_val_loss = infinity

─── EPOCH 0 ───────────────────────────────────────────
Step 11: trainer.train_epoch()
         → 1249 batches × 8 samples
         → For each batch: forward → loss → backward → clip → step
         → Returns: {"total": 2.14, "emotion": 1.35, "sentiment": 0.79}

Step 12: trainer.evaluate(val_loader)
         → 139 batches × 8 samples (no gradients)
         → Returns: losses + metrics (accuracy, precision)

Step 13: Append to metrics_data
Step 14: Print JSON metrics to stdout (SageMaker picks these up)
Step 15: Print GPU memory: "Peak GPU memory used: 3.42 GB"
Step 16: 2.14 < inf → Save model.pth, best_val_loss = 2.14

─── EPOCH 1 ───────────────────────────────────────────
Step 17-22: Same as Steps 11-16 with updated weights
         → train_loss: 1.87, val_loss: 1.95
         → 1.95 < 2.14 → Save model.pth, best_val_loss = 1.95

─── EPOCH 2 ───────────────────────────────────────────
Step 23-28: Same as Steps 11-16
         → train_loss: 1.63, val_loss: 1.78
         → 1.78 < 1.95 → Save model.pth, best_val_loss = 1.78

─── POST-TRAINING ─────────────────────────────────────
Step 29: "Evaluating on test set..."
Step 30: trainer.evaluate(test_loader, phase="test")
         → 327 batches × 8 samples
         → Returns test losses and metrics

Step 31: Print test JSON metrics to stdout
Step 32: Program exits normally (code 0)
```

---

## 8. Relationship with Other Modules

```
┌─────────────────────────────────────────────────────────────────────┐
│                          train.py (YOU ARE HERE)                     │
│                                                                     │
│  1. install_ffmpeg()        ──────→ install_ffmpeg.py               │
│     Ensures FFmpeg exists            Returns True/False             │
│                                                                     │
│  2. prepare_dataloaders()   ──────→ meld_dataset.py                 │
│     CSV + Videos → DataLoaders       MELDDataset class              │
│     │                                │                              │
│     │                                ├─ _load_video_frames()        │
│     │                                │   Uses: cv2 (OpenCV)         │
│     │                                │   Output: (30, 3, 224, 224)  │
│     │                                │                              │
│     │                                ├─ _extract_audio_features()   │
│     │                                │   Uses: FFmpeg + torchaudio  │
│     │                                │   Output: (1, 64, 300)       │
│     │                                │                              │
│     │                                └─ BERT tokenizer              │
│     │                                    Output: (128,) token ids   │
│     │                                                               │
│  3. MultimodalSentimentModel ─────→ models.py                       │
│     Neural network                   │                              │
│     │                                ├─ TextEncoder (BERT)          │
│     │                                │   768 → 128 features         │
│     │                                │                              │
│     │                                ├─ VideoEncoder (R3D-18)       │
│     │                                │   video → 128 features       │
│     │                                │                              │
│     │                                ├─ AudioEncoder (Conv1d)       │
│     │                                │   mel spec → 128 features    │
│     │                                │                              │
│     │                                ├─ FusionLayer                 │
│     │                                │   384 → 256 features         │
│     │                                │                              │
│     │                                ├─ EmotionClassifier           │
│     │                                │   256 → 7 classes            │
│     │                                │                              │
│     │                                └─ SentimentClassifier         │
│     │                                    256 → 3 classes            │
│     │                                                               │
│  4. MultimodalTrainer       ─────→ models.py                        │
│     Training orchestration           │                              │
│                                      ├─ Adam optimizer              │
│                                      ├─ ReduceLROnPlateau scheduler │
│                                      ├─ CrossEntropyLoss (×2)       │
│                                      ├─ TensorBoard logging         │
│                                      ├─ train_epoch()               │
│                                      └─ evaluate()                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Frequently Asked Questions

### Q: Why is `learning-rate` parsed but never used in `main()`?

The `--learning-rate` argument is parsed by `argparse` but the actual learning rates are hardcoded inside `MultimodalTrainer.__init__()` with per-layer rates (8e-6, 8e-5, 5e-4). To use the CLI argument, you would need to pass `args.learning_rate` to the trainer.

### Q: What happens if a video file is missing?

`MELDDataset.__getitem__()` returns `None`, which is filtered out by `collate_fn`. The batch will have fewer than `batch_size` samples but training continues.

### Q: Does this save the best model or the last model?

The **best model** (lowest validation loss) is saved via checkpointing in [Section 5m](#5m-best-model-checkpointing). However, the **test evaluation** in [Section 5n](#5n-test-set-evaluation) evaluates the **last epoch's model** (which may not be the best).

### Q: What happens on CPU (no GPU)?

Everything works the same but slower. GPU memory tracking is skipped. `device = torch.device('cpu')`.

### Q: What does `label_smoothing=0.05` do?

Instead of hard labels `[0, 0, 1, 0, 0, 0, 0]` (one-hot), uses soft labels `[0.007, 0.007, 0.957, 0.007, 0.007, 0.007, 0.007]`. This prevents the model from becoming overconfident and improves generalization.

### Q: How large is the model?

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| TextEncoder (BERT) | ~110M | Frozen (only 128×768 projection) |
| VideoEncoder (R3D-18) | ~33M | Frozen (only 128×512 final FC) |
| AudioEncoder (Conv1d) | ~25K | Frozen (only 128×128 projection) |
| FusionLayer | ~99K | ✓ Trainable |
| EmotionClassifier | ~17K | ✓ Trainable |
| SentimentClassifier | ~17K | ✓ Trainable |
| **Total** | **~143M** | **~230K trainable** |

