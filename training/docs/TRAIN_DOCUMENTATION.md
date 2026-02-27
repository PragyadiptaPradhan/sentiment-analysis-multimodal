# Train.py Documentation - Line by Line Explanation

## Table of Contents
1. [Imports](#imports)
2. [AWS SageMaker Configuration](#aws-sagemaker-configuration)
3. [Environment Setup](#environment-setup)
4. [Functions](#functions)
5. [Main Execution Flow](#main-execution-flow)

---

## Imports

### Standard Library Imports
```python
import os
import argparse
import sys
```

| Import | Purpose | Example |
|--------|---------|---------|
| `os` | Operating system interactions (file paths, environment variables) | `os.environ.get('SM_MODEL_DIR', ".")` - Get environment variable |
| `argparse` | Parse command-line arguments | Create CLI argument parser for epochs, batch size, etc. |
| `sys` | System-specific parameters and functions | `sys.exit(1)` - Exit program with error code |

### PyTorch Audio and ML Libraries
```python
import torchaudio
import torch
from tqdm import tqdm
```

| Import | Purpose | Example |
|--------|---------|---------|
| `torchaudio` | Audio processing for PyTorch | `torchaudio.list_audio_backends()` - List available audio backends |
| `torch` | PyTorch deep learning framework | `torch.device('cuda')` - Check for GPU availability |
| `tqdm` | Progress bar for loops | `for epoch in tqdm(range(20))` - Show training progress |

### Local Imports
```python
from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg
```

| Import | Source | Purpose |
|--------|--------|---------|
| `prepare_dataloaders` | meld_dataset.py | Creates train, validation, and test data loaders from CSV files |
| `MultimodalSentimentModel` | models.py | Neural network model for sentiment & emotion analysis |
| `MultimodalTrainer` | models.py | Training and evaluation class |
| `install_ffmpeg` | install_ffmpeg.py | Installs FFmpeg for audio/video processing |

---

## AWS SageMaker Configuration

### Environment Variable Setup
```python
# AWS SageMaker default paths
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', "/opt/ml/input/data/test")
```

**Explanation:**
- `os.environ.get(key, default)` retrieves environment variables with fallback defaults
- These variables are set by AWS SageMaker when running training jobs
- Falls back to local paths if running locally

**Example Usage:**
```
Running locally: SM_MODEL_DIR defaults to "."
Running on SageMaker: SM_MODEL_DIR = "/opt/ml/model" (set by SageMaker)
```

### GPU Memory Configuration
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
```

**Explanation:**
- Enables expandable GPU memory segments
- Prevents CUDA out-of-memory errors during training
- Allows dynamic memory allocation instead of pre-allocating all memory

**Example:**
```
Without: GPU reserves all memory upfront (could fail if not enough)
With: GPU allocates memory on-demand, expanding as needed
```

---

## Functions

### 1. `parse_args()` Function

**Purpose:** Parse command-line arguments for training configuration

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)
    
    return parser.parse_args()
```

**Line-by-Line Breakdown:**

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `parser = argparse.ArgumentParser()` | Creates argument parser object |
| 2 | `parser.add_argument("--epochs", type=int, default=20)` | Accepts epochs parameter, stores as integer, defaults to 20 |
| 3 | `parser.add_argument("--batch-size", type=int, default=16)` | Batch size parameter, defaults to 16 |
| 4 | `parser.add_argument("--learning-rate", type=float, default=0.001)` | Learning rate (float), defaults to 0.001 |
| 5 | `parser.add_argument("--train-dir", ...)` | Training data directory path |
| 6 | `return parser.parse_args()` | Parse and return all arguments |

**Usage Examples:**

```bash
# Example 1: Use defaults
python train.py

# Example 2: Custom training parameters
python train.py --epochs 50 --batch-size 32 --learning-rate 0.0001

# Example 3: Custom data directories
python train.py --train-dir /path/to/train --model-dir /path/to/save/model

# Result object:
# args.epochs = 50
# args.batch_size = 32
# args.learning_rate = 0.0001
# args.train_dir = "/path/to/train"
```

---

### 2. `main()` Function

**Purpose:** Orchestrate the entire training pipeline

#### Part A: FFmpeg Installation Check
```python
if not install_ffmpeg():
    print("Error: FFmpeg installation failed. Cannot continue training.")
    sys.exit(1)
```

**Explanation:**
- Checks if FFmpeg installation was successful
- FFmpeg is required for audio/video decoding
- Exits with error code 1 if installation fails
- Prevents training from starting without FFmpeg

**Example:**
```
If FFmpeg not found:
- Attempts installation
- If successful: Continue
- If fails: Exit program with error message
```

#### Part B: Audio Backend Information
```python
print("Available audio backends:")
print(str(torchaudio.list_audio_backends()))
```

**Explanation:**
- Lists available audio backends (e.g., "sox", "libsndfile", "soundfile")
- Useful for debugging audio processing issues
- Shows which backends PyTorch can use

**Example Output:**
```
Available audio backends:
['soundfile', 'sox']
```

#### Part C: Argument Parsing & Device Setup
```python
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Line Breakdown:**
- `args = parse_args()`: Get command-line arguments
- `device = torch.device(...)`: Select GPU (CUDA) if available, else CPU
- Ensures model runs on optimal hardware

**Example:**
```python
# If GPU available
device = torch.device('cuda')  # Use GPU

# If no GPU
device = torch.device('cpu')   # Use CPU

# Usage: model.to(device) - Move model to selected device
```

#### Part D: GPU Memory Tracking
```python
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    memory_used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Initial GPU memory used: {memory_used:.2f} GB")
```

**Line-by-Line:**

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `if torch.cuda.is_available()` | Check if GPU is available |
| 2 | `torch.cuda.reset_peak_memory_stats()` | Reset GPU memory statistics |
| 3 | `memory_used = torch.cuda.max_memory_allocated() / 1024**3` | Get memory in GB (divide bytes by 1024³) |
| 4 | `print(f"Initial GPU memory used: {memory_used:.2f} GB")` | Display memory (2 decimal places) |

**Example:**
```
Initial GPU memory used: 0.45 GB
```

#### Part E: Data Loading
```python
train_loader, val_loader, test_loader = prepare_dataloaders(
    train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
    train_video_dir=os.path.join(args.train_dir, 'train_splits'),
    dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
    dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
    test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
    test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
    batch_size=args.batch_size
)
```

**Explanation:**
- `prepare_dataloaders()`: Creates PyTorch DataLoaders from CSV files and video directories
- `os.path.join()`: Creates platform-independent file paths
- Returns 3 DataLoaders (train, validation, test) for batch processing

**Example Data Structure:**
```
Train Loader:
├── Batch 1: 16 video samples with labels
├── Batch 2: 16 video samples with labels
└── ... (continues for entire training set)

Each sample contains:
- Video frames (multi-modal data)
- Emotion label (anger, joy, sadness, neutral)
- Sentiment label (positive, negative, neutral)
```

#### Part F: Model & Trainer Initialization
```python
model = MultimodalSentimentModel().to(device)
trainer = MultimodalTrainer(model, train_loader, val_loader)
best_val_loss = float('inf')
```

**Explanation:**
- `MultimodalSentimentModel()`: Creates neural network instance
- `.to(device)`: Moves model to GPU or CPU
- `MultimodalTrainer()`: Creates trainer object with model and data loaders
- `best_val_loss = float('inf')`: Initialize to infinity to track best validation loss

**Example:**
```
Model initialized on GPU
best_val_loss = infinity (will be updated during training)
```

#### Part G: Metrics Dictionary
```python
metrics_data = {
    "train_losses": [],
    "val_losses": [],
    "epochs": []
}
```

**Explanation:**
- Initializes empty lists to track metrics
- Data structure for storing training history

**Example:**
```
After training:
metrics_data = {
    "train_losses": [0.85, 0.72, 0.68, ...],
    "val_losses": [0.90, 0.78, 0.74, ...],
    "epochs": [0, 1, 2, ...]
}
```

#### Part H: Training Loop
```python
for epoch in tqdm(range(args.epochs), desc="Epochs"):
    train_loss = trainer.train_epoch()
    val_loss, val_metrics = trainer.evaluate(val_loader)
```

**Line Breakdown:**

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `for epoch in tqdm(range(args.epochs), desc="Epochs")` | Loop through epochs with progress bar |
| 2 | `train_loss = trainer.train_epoch()` | Train model for one epoch, get loss |
| 3 | `val_loss, val_metrics = trainer.evaluate(val_loader)` | Evaluate on validation set |

**Example Output:**
```
Epochs: 45%|████▌     | 9/20 [02:34<03:08, 18.69s/it]

Returns:
train_loss = {"total": 0.62, "emotion": 0.35, "sentiment": 0.27}
val_loss = {"total": 0.68, "emotion": 0.40, "sentiment": 0.28}
val_metrics = {"emotion_accuracy": 0.82, "sentiment_accuracy": 0.79, ...}
```

#### Part I: Metrics Recording
```python
metrics_data["train_losses"].append(train_loss["total"])
metrics_data["val_losses"].append(val_loss["total"])
metrics_data["epochs"].append(epoch)
```

**Explanation:**
- Appends current epoch's losses to tracking lists
- Maintains training history for analysis/plotting

**Example:**
```
Epoch 0: metrics_data["train_losses"] = [0.85]
Epoch 1: metrics_data["train_losses"] = [0.85, 0.72]
Epoch 2: metrics_data["train_losses"] = [0.85, 0.72, 0.68]
```

#### Part J: SageMaker Metrics Logging
```python
print(json.dumps({
    "metrics": [
        {"Name": "train:loss", "Value": train_loss["total"]},
        {"Name": "validation:loss", "Value": val_loss["total"]},
        {"Name": "validation:emotion_precision", "Value": val_metrics["emotion_precision"]},
        {"Name": "validation:emotion_accuracy", "Value": val_metrics["emotion_accuracy"]},
        {"Name": "validation:sentiment_precision", "Value": val_metrics["sentiment_precision"]},
        {"Name": "validation:sentiment_accuracy", "Value": val_metrics["sentiment_accuracy"]},
    ]
}))
```

**Explanation:**
- Outputs metrics in JSON format for SageMaker CloudWatch monitoring
- Allows tracking metrics in AWS console during training
- Enables automated monitoring and alerts

**Example Output:**
```json
{
  "metrics": [
    {"Name": "train:loss", "Value": 0.62},
    {"Name": "validation:loss", "Value": 0.68},
    {"Name": "validation:emotion_precision", "Value": 0.85},
    {"Name": "validation:emotion_accuracy", "Value": 0.82},
    {"Name": "validation:sentiment_precision", "Value": 0.78},
    {"Name": "validation:sentiment_accuracy", "Value": 0.79}
  ]
}
```

#### Part K: GPU Memory Monitoring During Training
```python
if torch.cuda.is_available():
    memory_used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory used: {memory_used:.2f} GB")
```

**Explanation:**
- Monitors GPU memory usage each epoch
- Helps identify memory leaks or bottlenecks
- Useful for model optimization

**Example:**
```
Epoch 0: Peak GPU memory used: 2.34 GB
Epoch 1: Peak GPU memory used: 2.35 GB
Epoch 2: Peak GPU memory used: 2.36 GB (indicates potential memory leak if growing linearly)
```

#### Part L: Model Checkpoint Saving
```python
if val_loss["total"] < best_val_loss:
    best_val_loss = val_loss["total"]
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
```

**Line Breakdown:**

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `if val_loss["total"] < best_val_loss` | Check if current validation loss is better |
| 2 | `best_val_loss = val_loss["total"]` | Update best loss record |
| 3 | `torch.save(model.state_dict(), ...)` | Save model weights to file |

**Example:**
```
Epoch 0: val_loss = 0.85 → Save model (0.85 < inf ✓)
Epoch 1: val_loss = 0.72 → Save model (0.72 < 0.85 ✓)
Epoch 2: val_loss = 0.75 → Skip (0.75 > 0.72 ✗)
Epoch 3: val_loss = 0.68 → Save model (0.68 < 0.72 ✓)
```

#### Part M: Post-Training Test Evaluation
```python
print("Evaluating on test set...")
test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
metrics_data["test_loss"] = test_loss["total"]
```

**Explanation:**
- Evaluates trained model on test set
- Provides final performance metrics
- Stores test loss in metrics dictionary

**Example:**
```
Evaluating on test set...
Test Loss: 0.71
```

#### Part N: Test Metrics Logging
```python
print(json.dumps({
    "metrics": [
        {"Name": "test:loss", "Value": test_loss["total"]},
        {"Name": "test:emotion_accuracy", "Value": test_metrics["emotion_accuracy"]},
        {"Name": "test:sentiment_accuracy", "Value": test_metrics["sentiment_accuracy"]},
        {"Name": "test:emotion_precision", "Value": test_metrics["emotion_precision"]},
        {"Name": "test:sentiment_precision", "Value": test_metrics["sentiment_precision"]},
    ]
}))
```

**Explanation:**
- Logs final test metrics in SageMaker format
- Provides comprehensive test performance evaluation

**Example Output:**
```json
{
  "metrics": [
    {"Name": "test:loss", "Value": 0.71},
    {"Name": "test:emotion_accuracy", "Value": 0.79},
    {"Name": "test:sentiment_accuracy", "Value": 0.76},
    {"Name": "test:emotion_precision", "Value": 0.77},
    {"Name": "test:sentiment_precision", "Value": 0.75}
  ]
}
```

---

## Main Execution Flow

### Entry Point
```python
if __name__ == "__main__":
    main()
```

**Explanation:**
- Only executes when script runs directly (not when imported)
- Calls main() function to start training pipeline

### Complete Training Workflow

```
1. Parse command-line arguments
   └─ epochs=20, batch_size=16, learning_rate=0.001

2. Install FFmpeg (if needed)
   └─ Required for audio/video decoding

3. Check available audio backends
   └─ Print supported backends for debugging

4. Setup device (GPU or CPU)
   └─ Auto-detect CUDA availability

5. Load data
   ├─ Train data: train_sent_emo.csv + train_splits/
   ├─ Val data: dev_sent_emo.csv + dev_splits_complete/
   └─ Test data: test_sent_emo.csv + output_repeated_splits_test/

6. Initialize model and trainer
   └─ Create MultimodalSentimentModel on device

7. Training loop (for each epoch):
   ├─ Run trainer.train_epoch()
   ├─ Evaluate on validation set
   ├─ Log metrics to SageMaker
   ├─ Monitor GPU memory
   ├─ Save model if validation loss improves
   └─ Display progress with tqdm

8. Final test evaluation
   ├─ Evaluate best model on test set
   └─ Log final metrics

9. Output: model.pth saved to args.model_dir
```

### Complete Example Usage

```bash
# Basic training with defaults
python train.py

# Custom training configuration
python train.py \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.0005 \
    --train-dir ./dataset/train \
    --val-dir ./dataset/dev \
    --test-dir ./dataset/test \
    --model-dir ./saved_models

# Output structure
Training Summary:
├── Epochs: 0-99
├── Training losses: [0.85, 0.72, ..., 0.45]
├── Validation losses: [0.90, 0.78, ..., 0.52]
├── Best model: saved_models/model.pth
├── Train time: ~8 hours
└── Final test accuracy: 78-81%
```

---

## Key Concepts

### 1. Multimodal Learning
- Processes multiple data types: video, audio, text (emotions and sentiment)
- Model learns joint representations from different modalities

### 2. Batch Processing
- Processes 16 samples simultaneously (default batch size)
- Speeds up training with parallel GPU computation

### 3. Loss Functions
- Combined loss: Emotion loss + Sentiment loss
- Model optimizes both tasks simultaneously

### 4. Validation Strategy
- Evaluates on validation set each epoch
- Saves best performing model
- Prevents overfitting

### 5. AWS SageMaker Integration
- Environment variables for distributed training
- CloudWatch monitoring of metrics
- Automatic model artifact management

---

## Debugging Guide

| Issue | Solution |
|-------|----------|
| FFmpeg not found | `python -m pip install ffmpeg-python` |
| CUDA out of memory | Reduce batch size or enable `expandable_segments` |
| Metrics not appearing | Check JSON format in CloudWatch |
| Model not saving | Verify `args.model_dir` exists and is writable |
| Data loading errors | Check CSV format and video directory paths |

