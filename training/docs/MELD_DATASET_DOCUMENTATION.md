# meld_dataset.py Documentation - Line by Line Explanation

## Table of Contents
1. [Imports](#imports)
2. [Environment Configuration](#environment-configuration)
3. [MELDDataset Class](#melddataset-class)
   - [`__init__`](#1-__init__-constructor)
   - [`_load_video_frames`](#2-_load_video_frames-method)
   - [`_extract_audio_features`](#3-_extract_audio_features-method)
   - [`__len__`](#4-__len__-method)
   - [`__getitem__`](#5-__getitem__-method)
4. [Standalone Functions](#standalone-functions)
   - [`collate_fn`](#6-collate_fn-function)
   - [`prepare_dataloaders`](#7-prepare_dataloaders-function)
5. [Main Block](#main-block)
6. [Data Flow Diagram](#data-flow-diagram)

---

## Imports

```python
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio
import soundfile as sf
```

| Import | Library | Purpose | Example |
|--------|---------|---------|---------|
| `Dataset` | PyTorch | Base class for custom datasets | `class MELDDataset(Dataset):` — inherit to create custom dataset |
| `DataLoader` | PyTorch | Batches and iterates over datasets | `DataLoader(dataset, batch_size=16)` — yields batches of 16 |
| `pd` | pandas | Read and manipulate CSV files | `pd.read_csv('data.csv')` — loads CSV into DataFrame |
| `AutoTokenizer` | HuggingFace Transformers | Tokenize text for BERT models | `tokenizer("Hello world")` → `{'input_ids': [101, 7592, ...]}` |
| `os` | Standard Library | File path operations | `os.path.join('/data', 'file.mp4')` → `'/data/file.mp4'` |
| `cv2` | OpenCV | Video/image processing | `cv2.VideoCapture('video.mp4')` — open video file |
| `np` | NumPy | Numerical array operations | `np.zeros((224, 224, 3))` — create blank image array |
| `torch` | PyTorch | Tensor operations and deep learning | `torch.FloatTensor([1.0, 2.0])` — create tensor |
| `subprocess` | Standard Library | Run external commands (FFmpeg) | `subprocess.run(["ffmpeg", ...])` — execute FFmpeg |
| `torchaudio` | PyTorch Audio | Audio transformations | `torchaudio.transforms.MelSpectrogram()` — create spectrogram |
| `sf` | SoundFile | Read/write audio files | `sf.read('audio.wav')` → `(waveform_array, sample_rate)` |

---

## Environment Configuration

```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

**Explanation:**
- Disables parallel tokenization in HuggingFace tokenizers
- Prevents deadlock warnings when using DataLoader with multiple workers
- Required because DataLoader already handles parallelism

**Example Warning (without this line):**
```
huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
Disabling parallelism to avoid deadlocks...
```

---

## MELDDataset Class

```python
class MELDDataset(Dataset):
```

**Explanation:**
- Inherits from PyTorch's `Dataset` base class
- Must implement `__len__()` and `__getitem__()` methods
- Represents the MELD (Multimodal EmotionLines Dataset) for emotion recognition

**Example:**
```python
# Creating a dataset instance
dataset = MELDDataset('train_sent_emo.csv', 'train_splits/')
print(len(dataset))  # Number of samples
sample = dataset[0]  # Get first sample
```

---

### 1. `__init__` (Constructor)

```python
def __init__(self, csv_path, video_dir):
    self.data = pd.read_csv(csv_path)
    self.video_dir = video_dir
    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    self.emotion_map = {
        'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
        'neutral': 4, 'sadness': 5, 'surprise': 6
    }
    self.sentiment_map = {
        'negative': 0, 'neutral': 1, 'positive': 2
    }
```

**Line-by-Line Breakdown:**

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `def __init__(self, csv_path, video_dir)` | Constructor takes CSV file path and video directory path |
| 2 | `self.data = pd.read_csv(csv_path)` | Load CSV into pandas DataFrame with all dialogue data |
| 3 | `self.video_dir = video_dir` | Store video directory path for later file lookups |
| 4 | `self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')` | Load pre-trained BERT tokenizer (30,522 vocab, lowercase) |
| 5 | `self.emotion_map = {...}` | Map emotion strings to integer labels (7 classes) |
| 6 | `self.sentiment_map = {...}` | Map sentiment strings to integer labels (3 classes) |

**Example CSV Structure (`train_sent_emo.csv`):**
```
| Sr No. | Utterance               | Speaker  | Emotion  | Sentiment | Dialogue_ID | Utterance_ID |
|--------|-------------------------|----------|----------|-----------|-------------|--------------|
| 1      | "How are you?"          | Rachel   | neutral  | neutral   | 0           | 0            |
| 2      | "I'm so happy!"        | Monica   | joy      | positive  | 0           | 1            |
| 3      | "That's terrible..."   | Ross     | sadness  | negative  | 1           | 0            |
```

**Example of tokenizer usage:**
```python
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
result = tokenizer("I'm so happy!")
# result = {
#   'input_ids': [101, 1045, 1005, 1049, 2061, 3407, 999, 102],
#   'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]
# }
# 101 = [CLS] token, 102 = [SEP] token
```

**Example of label mapping:**
```python
emotion_map['joy']       # → 3
emotion_map['anger']     # → 0
sentiment_map['positive'] # → 2
sentiment_map['negative'] # → 0
```

---

### 2. `_load_video_frames` Method

**Purpose:** Extract up to 30 frames from a video file, resize to 224×224, normalize, and return as a tensor.

```python
def _load_video_frames(self, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `cap = cv2.VideoCapture(video_path)` | Open video file for reading frames |
| 2 | `frames = []` | Initialize empty list to collect frames |

**Example:**
```python
cap = cv2.VideoCapture('dia0_utt1.mp4')
# cap is a VideoCapture object pointing to the video file
```

#### Validation Block
```python
try:
    if not cap.isOpened():
        raise ValueError(f"Video not found: {video_path}")
    
    ret, frame = cap.read()
    if not ret or frame is None:
        raise ValueError(f"Video not found : {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `if not cap.isOpened()` | Check if video file was opened successfully |
| 2 | `ret, frame = cap.read()` | Try reading first frame: `ret` = True/False, `frame` = pixel array |
| 3 | `if not ret or frame is None` | Validate that the frame was actually read |
| 4 | `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)` | Reset read position to frame 0 (since we read one frame for validation) |

**Example:**
```python
cap = cv2.VideoCapture('dia0_utt1.mp4')
cap.isOpened()  # True if file exists and is valid video

ret, frame = cap.read()
# ret = True, frame.shape = (360, 480, 3) — height, width, BGR channels

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Go back to start
```

#### Frame Extraction Loop
```python
    while len(frames) < 30 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frames.append(frame)
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `while len(frames) < 30 and cap.isOpened()` | Read up to 30 frames maximum |
| 2 | `ret, frame = cap.read()` | Read next frame from video |
| 3 | `if not ret: break` | Stop if no more frames |
| 4 | `frame = cv2.resize(frame, (224, 224))` | Resize frame to 224×224 pixels (standard for CNNs like ResNet) |
| 5 | `frame = frame / 255.0` | Normalize pixel values from [0, 255] to [0.0, 1.0] |
| 6 | `frames.append(frame)` | Add processed frame to list |

**Example:**
```python
# Original frame: shape (360, 480, 3), values [0-255]
frame = cap.read()[1]               # shape: (360, 480, 3)
frame = cv2.resize(frame, (224, 224))  # shape: (224, 224, 3)
frame = frame / 255.0               # values now [0.0 - 1.0]

# After loop: frames = [frame0, frame1, ..., frame29]
# Each frame shape: (224, 224, 3)
```

#### Cleanup and Release
```python
except Exception as e:
    raise ValueError(f"Video error: {str(e)}")

finally:
    cap.release()
```

| Line | Explanation |
|------|-------------|
| `except Exception as e` | Catch any errors during video processing |
| `cap.release()` | Always release the video file handle (even if error occurred) |

#### Padding or Truncating
```python
if(len(frames) == 0):
    raise ValueError(f"No frames extracted")

if(len(frames) < 30):
    frames += [np.zeros_like(frames[0])] * (30 - len(frames))
else:
    frames = frames[:30]
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `if len(frames) == 0` | Error if no frames were extracted |
| 2 | `frames += [np.zeros_like(frames[0])] * (30 - len(frames))` | Pad with black frames if fewer than 30 |
| 3 | `frames = frames[:30]` | Truncate to exactly 30 if more |

**Example:**
```python
# If video has 12 frames:
# frames = [frame0, frame1, ..., frame11, black, black, ..., black]
#           ├─── 12 real frames ───┤ ├──── 18 padding frames ────┤
# Total: 30 frames

# If video has 50 frames:
# frames = [frame0, frame1, ..., frame29]
# Total: 30 frames (truncated)
```

#### Tensor Conversion and Permute
```python
# Before permute: [frames, height, width, channels]
# After permute:  [frames, channels, height, width]
return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
```

**Explanation:**
- `np.array(frames)` — Convert list of frames to NumPy array: shape `(30, 224, 224, 3)`
- `torch.FloatTensor(...)` — Convert to PyTorch tensor
- `.permute(0, 3, 1, 2)` — Rearrange dimensions for PyTorch's expected format

**Example:**
```python
frames_array = np.array(frames)
# Shape: (30, 224, 224, 3) → [num_frames, height, width, channels]

tensor = torch.FloatTensor(frames_array).permute(0, 3, 1, 2)
# Shape: (30, 3, 224, 224) → [num_frames, channels, height, width]
# PyTorch convolutions expect: (batch, channels, height, width)
```

---

### 3. `_extract_audio_features` Method

**Purpose:** Extract audio from video, convert to mel spectrogram features.

#### Audio Path Setup
```python
def _extract_audio_features(self, video_path):
    audio_path = video_path.replace('.mp4', '.wav')
```

**Explanation:**
- Creates WAV file path from video path (temporary file)

**Example:**
```python
video_path = '/data/train_splits/dia0_utt1.mp4'
audio_path = '/data/train_splits/dia0_utt1.wav'
```

#### FFmpeg Audio Extraction
```python
subprocess.run([
    "ffmpeg",
    '-i', video_path,       # Input: video file
    '-vn',                   # No video (audio only)
    '-acodec', 'pcm_s16le', # Audio codec: 16-bit PCM
    '-ar', '16000',          # Sample rate: 16kHz
    '-ac', '1',              # Mono channel
    audio_path               # Output: WAV file
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

| Flag | Meaning | Example |
|------|---------|---------|
| `-i video_path` | Input file | `-i dia0_utt1.mp4` |
| `-vn` | Discard video stream | Only process audio |
| `-acodec pcm_s16le` | Audio codec: 16-bit signed little-endian PCM | Standard WAV format |
| `-ar 16000` | Resample audio to 16,000 Hz | Speech recognition standard |
| `-ac 1` | Convert to mono (1 channel) | Simplify processing |
| `check=True` | Raise exception if FFmpeg fails | Error handling |
| `stdout=subprocess.DEVNULL` | Suppress FFmpeg stdout output | Clean console |
| `stderr=subprocess.DEVNULL` | Suppress FFmpeg stderr output | Clean console |

**Example Command Equivalent:**
```bash
ffmpeg -i dia0_utt1.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 dia0_utt1.wav
```

#### Loading Audio with SoundFile
```python
waveform_np, sample_rate = sf.read(audio_path, dtype='float32')
```

**Explanation:**
- Reads WAV file into NumPy array
- `dtype='float32'` — Load as 32-bit float values
- Returns waveform data and sample rate

**Example:**
```python
waveform_np, sample_rate = sf.read('dia0_utt1.wav', dtype='float32')
# waveform_np.shape = (48000,) — 3 seconds at 16kHz
# sample_rate = 16000
# waveform_np values: [-0.023, 0.045, -0.012, ...]
```

#### Ensuring Correct Shape
```python
if waveform_np.ndim == 1:
    waveform_np = np.expand_dims(waveform_np, axis=0)
else:
    waveform_np = waveform_np.T
```

| Condition | Before | After | Explanation |
|-----------|--------|-------|-------------|
| Mono (1D) | `(48000,)` | `(1, 48000)` | Add channel dimension |
| Stereo (2D) | `(48000, 2)` | `(2, 48000)` | Transpose to [channels, time] |

**Example:**
```python
# Mono audio
waveform_np = np.array([0.1, 0.2, 0.3])       # shape: (3,)
waveform_np = np.expand_dims(waveform_np, 0)    # shape: (1, 3)

# Stereo audio
waveform_np = np.array([[0.1, 0.2], [0.3, 0.4]])  # shape: (2, 2) [time, channels]
waveform_np = waveform_np.T                          # shape: (2, 2) [channels, time]
```

#### Convert to Tensor and Resample
```python
waveform = torch.from_numpy(waveform_np)

if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `torch.from_numpy(waveform_np)` | Convert NumPy array to PyTorch tensor |
| 2 | `if sample_rate != 16000` | Check if resampling needed |
| 3 | `Resample(sample_rate, 16000)` | Create resampler from original rate to 16kHz |
| 4 | `waveform = resampler(waveform)` | Apply resampling |

**Example:**
```python
# If audio was 44100 Hz:
resampler = torchaudio.transforms.Resample(44100, 16000)
# Input:  tensor of shape (1, 44100) — 1 second at 44.1kHz
# Output: tensor of shape (1, 16000) — 1 second at 16kHz
```

#### Mel Spectrogram Computation
```python
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=512
)

mel_spec = mel_spectrogram(waveform)
```

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `sample_rate=16000` | 16 kHz | Input audio sample rate |
| `n_mels=64` | 64 | Number of mel frequency bands (vertical resolution) |
| `n_fft=1024` | 1024 | FFT window size (frequency resolution) |
| `hop_length=512` | 512 | Step between FFT windows (time resolution) |

**Example:**
```python
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
)

waveform = torch.randn(1, 48000)  # 3 seconds of audio at 16kHz
mel_spec = mel_spectrogram(waveform)
# mel_spec.shape = (1, 64, 94)
# Shape: [channels=1, mel_bands=64, time_frames=94]

# Visualization concept:
# ┌─────────────────────────────────┐
# │ High freq  ░░▓▓░░░░▓▓▓░░░░░░░░│ mel band 63
# │            ░░▓▓▓░░░▓▓▓▓░░░░░░░│
# │            ░▓▓▓▓▓░▓▓▓▓▓▓░░░░░░│
# │            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░│
# │ Low freq   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ mel band 0
# └─────────────────────────────────┘
#              Time →
```

#### Normalization
```python
mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
```

**Explanation:**
- Z-score normalization: centers data around 0 with standard deviation of 1
- Ensures consistent scale across all audio samples
- Formula: $z = \frac{x - \mu}{\sigma}$

**Example:**
```python
mel_spec = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
# mean = 3.0, std = 1.58
normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
# normalized = [[-1.26, -0.63, 0.0, 0.63, 1.26]]
```

#### Padding or Truncating to Fixed Length
```python
if mel_spec.size(2) < 300:
    padding = 300 - mel_spec.size(2)
    mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
else:
    mel_spec = mel_spec[:, :, :300]

return mel_spec
```

| Condition | Action | Explanation |
|-----------|--------|------------|
| Time frames < 300 | Pad with zeros on the right | Short audio gets zero-padded |
| Time frames >= 300 | Truncate to 300 | Long audio gets cut off |

**Example:**
```python
# Short audio (94 time frames):
mel_spec.shape  # (1, 64, 94)
padding = 300 - 94  # = 206
mel_spec = torch.nn.functional.pad(mel_spec, (0, 206))
mel_spec.shape  # (1, 64, 300)

# Long audio (500 time frames):
mel_spec.shape  # (1, 64, 500)
mel_spec = mel_spec[:, :, :300]
mel_spec.shape  # (1, 64, 300)

# Final output shape is always: (1, 64, 300)
# [channels, mel_bands, time_frames]
```

#### Error Handling and Cleanup
```python
except subprocess.CalledProcessError as e:
    raise ValueError(f"Audio extraction error: {str(e)}")
except Exception as e:
    raise ValueError(f"Audio error: {str(e)}")
finally:
    if os.path.exists(audio_path):
        os.remove(audio_path)
```

| Block | Purpose |
|-------|---------|
| `subprocess.CalledProcessError` | Catches FFmpeg failures (file not found, codec error) |
| `Exception` | Catches any other errors (SoundFile read error, shape mismatch) |
| `finally: os.remove(audio_path)` | Always delete the temporary WAV file to save disk space |

**Example:**
```python
# FFmpeg fails → "Audio extraction error: Command 'ffmpeg' returned non-zero exit status 1"
# File read fails → "Audio error: Error reading audio.wav"
# Cleanup: dia0_utt1.wav is deleted regardless of success/failure
```

---

### 4. `__len__` Method

```python
def __len__(self):
    return len(self.data)
```

**Explanation:**
- Returns the total number of samples in the dataset
- `self.data` is the pandas DataFrame loaded from CSV
- Required by PyTorch's `Dataset` class

**Example:**
```python
dataset = MELDDataset('train_sent_emo.csv', 'train_splits/')
len(dataset)  # 9989 (number of rows in train CSV)

# Used by DataLoader to know how many batches to create:
# With batch_size=16: 9989 / 16 ≈ 625 batches per epoch
```

---

### 5. `__getitem__` Method

**Purpose:** Load and process a single sample (text + video + audio + labels) by index.

#### Index Handling
```python
def __getitem__(self, idx):
    if isinstance(idx, torch.Tensor):
        idx = idx.item()
    row = self.data.iloc[idx]
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `if isinstance(idx, torch.Tensor)` | Check if index is a tensor (from DataLoader sampling) |
| 2 | `idx = idx.item()` | Convert tensor to Python int |
| 3 | `row = self.data.iloc[idx]` | Get the row at position `idx` from the DataFrame |

**Example:**
```python
# idx could be: 42 (int) or tensor(42) (from DataLoader)
idx = torch.tensor(42)
idx = idx.item()  # → 42

row = self.data.iloc[42]
# row = {
#   'Utterance': "I'm so happy!",
#   'Speaker': 'Monica',
#   'Emotion': 'joy',
#   'Sentiment': 'positive',
#   'Dialogue_ID': 0,
#   'Utterance_ID': 1
# }
```

#### Video File Path Construction
```python
video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
path = os.path.join(self.video_dir, video_filename)

video_path_exists = os.path.exists(path)

if video_path_exists == False:
    raise FileNotFoundError(f"No video found for filename: {path}")
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"` | Build filename from dialogue and utterance IDs |
| 2 | `os.path.join(self.video_dir, video_filename)` | Create full path to video file |
| 3 | `os.path.exists(path)` | Check if video file exists on disk |
| 4 | `raise FileNotFoundError(...)` | Error if video is missing |

**Example:**
```python
# row['Dialogue_ID'] = 5, row['Utterance_ID'] = 3
video_filename = "dia5_utt3.mp4"
path = "/data/train_splits/dia5_utt3.mp4"

os.path.exists(path)  # True or False
```

#### Text Tokenization
```python
text_inputs = self.tokenizer(row['Utterance'],
                             padding='max_length',
                             truncation=True,
                             max_length=128,
                             return_tensors='pt')
```

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `row['Utterance']` | e.g. `"I'm so happy!"` | The dialogue text to tokenize |
| `padding='max_length'` | — | Pad short sequences to `max_length` with zeros |
| `truncation=True` | — | Cut sequences longer than `max_length` |
| `max_length=128` | 128 tokens | Fixed sequence length |
| `return_tensors='pt'` | PyTorch | Return PyTorch tensors |

**Example:**
```python
text_inputs = tokenizer("I'm so happy!",
                        padding='max_length', truncation=True,
                        max_length=128, return_tensors='pt')

# text_inputs = {
#   'input_ids': tensor([[101, 1045, 1005, 1049, 2061, 3407, 999, 102, 0, 0, ..., 0]]),
#                        [CLS] I     '     m    so   happy  !   [SEP] [PAD] ...
#   Shape: (1, 128)
#
#   'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ..., 0]])
#                             real tokens = 1, padding = 0
#   Shape: (1, 128)
# }
```

#### Multimodal Feature Extraction
```python
video_frames = self._load_video_frames(path)
audio_features = self._extract_audio_features(path)
```

| Line | Output Shape | Description |
|------|-------------|-------------|
| `self._load_video_frames(path)` | `(30, 3, 224, 224)` | 30 frames, 3 channels, 224×224 pixels |
| `self._extract_audio_features(path)` | `(1, 64, 300)` | 1 channel, 64 mel bands, 300 time steps |

#### Label Mapping
```python
emotion_label = self.emotion_map.get(row['Emotion'].lower())
sentiment_label = self.sentiment_map.get(row['Sentiment'].lower())
```

**Example:**
```python
row['Emotion'] = 'Joy'
emotion_label = emotion_map.get('joy')  # → 3

row['Sentiment'] = 'Positive'
sentiment_label = sentiment_map.get('positive')  # → 2
```

#### Return Dictionary
```python
return {
    'text_inputs': {
        'input_ids': text_inputs['input_ids'].squeeze(),
        'attention_mask': text_inputs['attention_mask'].squeeze()
    },
    'video_frames': video_frames,
    'audio_features': audio_features,
    'emotion_labels': torch.tensor(emotion_label),
    'sentiment_labels': torch.tensor(sentiment_label)
}
```

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `text_inputs.input_ids` | `(128,)` | LongTensor | Token IDs (`.squeeze()` removes batch dim) |
| `text_inputs.attention_mask` | `(128,)` | LongTensor | 1 for real tokens, 0 for padding |
| `video_frames` | `(30, 3, 224, 224)` | FloatTensor | 30 video frames |
| `audio_features` | `(1, 64, 300)` | FloatTensor | Mel spectrogram |
| `emotion_labels` | `()` scalar | LongTensor | Emotion class index (0-6) |
| `sentiment_labels` | `()` scalar | LongTensor | Sentiment class index (0-2) |

**Example of `.squeeze()`:**
```python
tensor = torch.tensor([[1, 2, 3]])  # shape: (1, 3)
tensor.squeeze()                     # shape: (3,) — removes dimension of size 1
```

#### Error Handling
```python
except Exception as e:
    print(f"Error processing index {path}: {str(e)}")
    return None
```

**Explanation:**
- If any error occurs (video not found, audio extraction fails, etc.), returns `None`
- `None` values are filtered out later by `collate_fn`
- Prevents one bad sample from crashing the entire training run

**Example:**
```
Error processing index /data/train_splits/dia5_utt3.mp4: Video not found
→ Returns None (skipped during training)
```

---

## Standalone Functions

### 6. `collate_fn` Function

```python
def collate_fn(batch):
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `batch = list(filter(None, batch))` | Remove all `None` entries from the batch |
| 2 | `default_collate(batch)` | Stack remaining samples into batched tensors |

**Example:**
```python
# Input batch (batch_size=4, but one failed):
batch = [sample_1, None, sample_3, sample_4]

# After filter:
batch = [sample_1, sample_3, sample_4]  # 3 valid samples

# After default_collate: stacks into batch tensors
# {
#   'video_frames': tensor of shape (3, 30, 3, 224, 224),  ← 3 samples
#   'audio_features': tensor of shape (3, 1, 64, 300),
#   'emotion_labels': tensor of shape (3,),
#   ...
# }
```

**Why is this needed?**
- Some video files may be corrupted or missing
- `__getitem__` returns `None` for failed samples
- Without `collate_fn`, PyTorch's default collate would crash on `None`
- This gracefully skips bad samples

---

### 7. `prepare_dataloaders` Function

**Purpose:** Create train, validation, and test DataLoaders from CSV and video directories.

```python
def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size=32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)
```

| Line | Code | Explanation |
|------|------|-------------|
| 1 | `MELDDataset(train_csv, train_video_dir)` | Create training dataset object |
| 2 | `MELDDataset(dev_csv, dev_video_dir)` | Create validation (dev) dataset object |
| 3 | `MELDDataset(test_csv, test_video_dir)` | Create test dataset object |

#### DataLoader Creation
```python
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader
```

| Parameter | Train | Dev/Test | Explanation |
|-----------|-------|----------|-------------|
| `batch_size` | 32 (default) | 32 (default) | Number of samples per batch |
| `shuffle` | `True` | Not set (default `False`) | Randomize order each epoch (only for training) |
| `collate_fn` | `collate_fn` | `collate_fn` | Custom function to handle `None` samples |

**Why shuffle only training data?**
- Training: Random order prevents model from memorizing sequence patterns
- Validation/Test: Consistent order for reproducible evaluation

**Example:**
```python
train_loader, dev_loader, test_loader = prepare_dataloaders(
    train_csv='dataset/train/train_sent_emo.csv',
    train_video_dir='dataset/train/train_splits/',
    dev_csv='dataset/dev/dev_sent_emo.csv',
    dev_video_dir='dataset/dev/dev_splits_complete/',
    test_csv='dataset/test/test_sent_emo.csv',
    test_video_dir='dataset/test/output_repeated_splits_test/',
    batch_size=16
)

# train_loader: ~625 batches (9989 samples / 16)
# dev_loader:   ~69 batches (1109 samples / 16)
# test_loader:  ~170 batches (2610 samples / 16)

for batch in train_loader:
    print(batch['video_frames'].shape)   # (16, 30, 3, 224, 224)
    print(batch['audio_features'].shape) # (16, 1, 64, 300)
    print(batch['emotion_labels'].shape) # (16,)
    break
```

---

## Main Block

```python
if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits/',
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete/',
        '../dataset/test/test_sent_emo.csv', '../dataset/test/output_repeated_splits_test/',
        batch_size=4
    )
    
    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break
```

**Explanation:**
- Only runs when executing `python meld_dataset.py` directly
- Creates all DataLoaders with small batch size (4) for testing
- Prints one batch and stops (quick sanity check)
- Uses relative paths (`../dataset/`) assuming script is run from `training/` directory

**Example Output:**
```
# text_inputs:
{'input_ids': tensor([[101, 2129, ..., 0],
                      [101, 1045, ..., 0],
                      [101, 2008, ..., 0],
                      [101, 2339, ..., 0]]),
 'attention_mask': tensor([[1, 1, ..., 0],
                           [1, 1, ..., 0],
                           [1, 1, ..., 0],
                           [1, 1, ..., 0]])}

# video_frames:
torch.Size([4, 30, 3, 224, 224])  # 4 samples, 30 frames each

# audio_features:
torch.Size([4, 1, 64, 300])  # 4 samples, mel spectrograms

# emotion_label:
tensor([4, 3, 0, 6])  # neutral, joy, anger, surprise

# sentiment_label:
tensor([1, 2, 0, 2])  # neutral, positive, negative, positive
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        meld_dataset.py                               │
│                                                                      │
│  CSV File (train_sent_emo.csv)                                       │
│  ┌──────────────────────────────────────────────┐                    │
│  │ Utterance | Emotion | Sentiment | Dia_ID | Utt_ID │               │
│  │ "Hello!"  | joy     | positive  | 0      | 1      │               │
│  └──────────────────────────────────────────────┘                    │
│         │                    │                     │                  │
│         ▼                    ▼                     ▼                  │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐         │
│  │ BERT        │   │ Label Maps   │   │ Video File Lookup │         │
│  │ Tokenizer   │   │              │   │ dia0_utt1.mp4     │         │
│  │             │   │ joy → 3      │   │                   │         │
│  │ "Hello!" →  │   │ positive → 2 │   │    ┌──────┐       │         │
│  │ [101,7592,  │   │              │   │    │ Video│       │         │
│  │  ...,102]   │   │              │   │    └──┬───┘       │         │
│  └─────┬───────┘   └──────┬───────┘   │       │          │         │
│        │                  │            │   ┌───┴────┐     │         │
│        │                  │            │   ▼        ▼     │         │
│        │                  │            │ Video    Audio   │         │
│        │                  │            │ Frames   Extract │         │
│        │                  │            │ (OpenCV) (FFmpeg)│         │
│        │                  │            │   │        │     │         │
│        │                  │            │   ▼        ▼     │         │
│        │                  │            │ (30,3,   (1,64,  │         │
│        │                  │            │  224,224) 300)   │         │
│        │                  │            └───┬────────┬─────┘         │
│        │                  │                │        │               │
│        ▼                  ▼                ▼        ▼               │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │                  Sample Dictionary                       │        │
│  │  {                                                       │        │
│  │    'text_inputs': {input_ids: (128,),                    │        │
│  │                    attention_mask: (128,)},               │        │
│  │    'video_frames': (30, 3, 224, 224),                    │        │
│  │    'audio_features': (1, 64, 300),                       │        │
│  │    'emotion_labels': tensor(3),                          │        │
│  │    'sentiment_labels': tensor(2)                         │        │
│  │  }                                                       │        │
│  └──────────────────────────────┬──────────────────────────┘        │
│                                 │                                    │
│                                 ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │               DataLoader (batch_size=16)                 │        │
│  │  collate_fn: filters None, stacks into batch tensors     │        │
│  │                                                          │        │
│  │  Output batch:                                           │        │
│  │    video_frames:  (16, 30, 3, 224, 224)                  │        │
│  │    audio_features: (16, 1, 64, 300)                      │        │
│  │    emotion_labels: (16,)                                 │        │
│  │    sentiment_labels: (16,)                               │        │
│  └─────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Summary

| Concept | Implementation | Why |
|---------|---------------|-----|
| **Fixed frame count** | Pad/truncate to 30 frames | Consistent tensor sizes for batching |
| **Fixed audio length** | Pad/truncate to 300 time steps | Consistent tensor sizes for batching |
| **Fixed text length** | Pad/truncate to 128 tokens | Consistent tensor sizes for batching |
| **Image normalization** | Divide by 255.0 | Scale pixels to [0, 1] for neural networks |
| **Audio normalization** | Z-score (mean=0, std=1) | Consistent scale for neural networks |
| **None handling** | `collate_fn` filters None | Skip corrupted files without crashing |
| **Temporary WAV files** | Create then delete in `finally` | Save disk space, only video files stored |
| **BERT tokenizer** | Pre-trained `bert-base-uncased` | Leverages pre-trained language understanding |

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|---------|
| `Video not found: ...` | Video file doesn't exist at constructed path | Check `Dialogue_ID` and `Utterance_ID` match filenames |
| `No frames extracted` | Video exists but is empty/corrupted | Re-download or skip the video |
| `Audio extraction error` | FFmpeg failed to extract audio | Ensure FFmpeg is installed and video has audio track |
| `FileNotFoundError` | CSV references non-existent video | Verify dataset integrity |
| `TOKENIZERS_PARALLELISM` warning | Tokenizer forking issues | Already fixed with `os.environ` setting |
