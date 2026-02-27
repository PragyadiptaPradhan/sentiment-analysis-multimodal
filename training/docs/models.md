# models.py – Line‑by‑Line Explanation with Examples

This document explains every line of `models.py` and shows how each class or function is used, with concrete examples.

---

## Imports and Basic Setup

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

- `import os` – Imports Python's standard `os` module, used to check environment variables and construct paths (e.g., deciding where to store TensorBoard logs).
- `import torch.nn as nn` – Imports PyTorch's neural network module and aliases it as `nn` for convenient access to layers like `Linear`, `Conv1d`, `ReLU`, etc.
- `from transformers import BertModel` – Imports the `BertModel` class from the Hugging Face `transformers` library to use a pretrained BERT encoder for text.
- `from torchvision import models as vision_models` – Imports computer vision models from `torchvision` and aliases them as `vision_models` (used to get the 3D ResNet).
- `import torch` – Imports the main PyTorch package; needed for tensors, optimizers, utility functions, etc.
- `from sklearn.metrics import precision_score, accuracy_score` – Imports evaluation metrics from scikit-learn for computing precision and accuracy.
- `from torch.utils.tensorboard import SummaryWriter` – Imports `SummaryWriter` to log metrics to TensorBoard.
- `from datetime import datetime` – Imports `datetime` to generate timestamped run directories.
- `from meld_dataset import MELDDataset` – Imports a custom PyTorch dataset class for the MELD dataset, used in the example under the `__main__` block.

**Example usage:**

```python
from training.models import MultimodalSentimentModel

model = MultimodalSentimentModel()
```

This snippet relies on all of the imports above being available.

---

## TextEncoder

```python
class TextEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)
    
    def forward(self, input_ids, attention_mask):
        # Extract BERT features
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)

        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output

        return self.projection(pooled_output)
```

- `class TextEncoder(nn.Module):` – Defines a neural network module for encoding text; it inherits from `nn.Module` so it can be used inside larger PyTorch models.
- `def __init__(self):` – Constructor where the submodules and configuration are defined.
- `super().__init__()` – Calls the base `nn.Module` constructor to correctly register parameters and buffers.
- `self.bert = BertModel.from_pretrained('bert-base-uncased')` – Loads a pretrained BERT model with uncased vocabulary from Hugging Face; this model outputs contextual embeddings for each token and a pooled `[CLS]` vector.
- `for param in self.bert.parameters():` – Iterates through every parameter in the BERT model.
- `param.requires_grad = False` – Freezes each BERT parameter so it is not updated during training (only downstream layers train).
- `self.projection = nn.Linear(768, 128)` – Defines a linear layer to project the 768‑dimensional BERT pooled output into a 128‑dimensional feature space.
- `def forward(self, input_ids, attention_mask):` – Defines how input tensors are transformed when the module is called.
- `# Extract BERT features` – Comment explaining that the next line feeds data to BERT.
- `outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)` – Runs the BERT model on token IDs and attention mask, returning a named tuple that includes `last_hidden_state` and `pooler_output`.
- `# Use the [CLS] token representation` – Comment indicating we will use the pooled representation corresponding to the `[CLS]` token.
- `pooled_output = outputs.pooler_output` – Extracts the `[CLS]` pooled embedding (shape `(batch_size, 768)`).
- `return self.projection(pooled_output)` – Projects the pooled embedding into 128‑dimensional features and returns them.

**Example usage:**

```python
encoder = TextEncoder()
dummy_input_ids = torch.randint(0, 30522, (2, 10))      # batch=2, seq_len=10
dummy_attention_mask = torch.ones_like(dummy_input_ids)  # keep all tokens

text_features = encoder(dummy_input_ids, dummy_attention_mask)
print(text_features.shape)  # torch.Size([2, 128])
```

---

## VideoEncoder

```python
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained = True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width] 
        x = x.transpose(1,2)
        return self.backbone(x)
```

- `class VideoEncoder(nn.Module):` – Defines a neural network module for encoding video sequences.
- `def __init__(self):` – Constructor for the video encoder.
- `super().__init__()` – Initializes the `nn.Module` base class.
- `self.backbone = vision_models.video.r3d_18(pretrained = True)` – Loads a pretrained 3D ResNet‑18 model for video classification from `torchvision`.
- `for param in self.backbone.parameters():` – Iterates through all parameters of the 3D ResNet backbone.
- `param.requires_grad = False` – Freezes all backbone parameters so only the new head is trained.
- `num_ftrs = self.backbone.fc.in_features` – Gets the number of input features to the original fully‑connected classification layer so we can replace it properly.
- `self.backbone.fc = nn.Sequential(... )` – Replaces the original classification head with a new head that outputs a 128‑dimensional feature vector.
  - `nn.Linear(num_ftrs, 128)` – Linear layer mapping backbone features to 128 dimensions.
  - `nn.ReLU()` – Non‑linear activation.
  - `nn.Dropout(0.2)` – Regularization by randomly dropping 20% of neurons during training.
- `def forward(self, x):` – Defines how input video tensors are transformed.
- Comment `# [batch_size, frames, channels, height, width] -> ...` – Documents that the input has frames before channels, but the backbone expects channels before frames.
- `x = x.transpose(1,2)` – Swaps the second and third dimensions to go from `(B, T, C, H, W)` to `(B, C, T, H, W)`.
- `return self.backbone(x)` – Passes the permuted video batch through the 3D ResNet backbone and returns the resulting 128‑dimensional features.

**Example usage:**

```python
encoder = VideoEncoder()
dummy_video = torch.randn(2, 8, 3, 112, 112)  # batch=2, frames=8

video_features = encoder(dummy_video)
print(video_features.shape)  # torch.Size([2, 128])
```

---

## AudioEncoder

```python
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1d = nn.Sequential(
            # lower Level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2),

            #Higer level freatures
            nn.Conv1d(64, 128, kernel_size = 3),
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
    def forward(self, x):
        x = x.squeeze(1)
        features = self.Conv1d(x)

        return self.projection(features.squeeze(-1))
```

- `class AudioEncoder(nn.Module):` – Defines a neural network module for encoding audio features.
- `def __init__(self):` – Constructor for the audio encoder.
- `super().__init__()` – Initializes the base `nn.Module`.
- `self.Conv1d = nn.Sequential(... )` – Builds a small 1D CNN feature extractor as a sequence of layers.
  - `nn.Conv1d(64, 64, kernel_size=3)` – First convolutional layer that maintains 64 channels and uses a kernel width of 3 over time.
  - `nn.BatchNorm1d(64)` – Normalizes the 64 channels to stabilize training.
  - `nn.ReLU()` – Applies ReLU non‑linearity.
  - `nn.MaxPool1d(kernel_size = 2)` – Reduces the temporal resolution by taking max over window size 2.
  - Second conv block: `nn.Conv1d(64, 128, kernel_size = 3)`, `nn.BatchNorm1d(128)`, `nn.ReLU()` – Increases channels to 128 and extracts higher‑level temporal features.
  - `nn.AdaptiveAvgPool1d(1)` – Aggregates features over time into a single time step, producing shape `(batch, 128, 1)`.
- `for param in self.Conv1d.parameters():` – Iterates over all parameters in the convolutional stack.
- `param.requires_grad = False` – Freezes these convolutional layers so only the projection is trained.
- `self.projection = nn.Sequential(... )` – Defines a small MLP to project the pooled 128‑channel representation.
  - `nn.Linear(128, 128)` – Linear mapping from 128 to 128 dimensions.
  - `nn.ReLU()` – Non‑linear activation.
  - `nn.Dropout(0.2)` – Regularization.
- `def forward(self, x):` – Forward pass definition.
- `x = x.squeeze(1)` – Removes a singleton dimension (e.g., going from `(B, 1, 64, T)` to `(B, 64, T)`) to match `Conv1d`'s expected input.
- `features = self.Conv1d(x)` – Passes the audio features through the frozen convolutional stack.
- `return self.projection(features.squeeze(-1))` – Squeezes the last dimension (from `(B, 128, 1)` to `(B, 128)`) and applies the projection MLP, returning `(batch_size, 128)`.

**Example usage:**

```python
encoder = AudioEncoder()
dummy_audio = torch.randn(2, 1, 64, 100)  # (B, 1, channels=64, time=100)

audio_features = encoder(dummy_audio)
print(audio_features.shape)  # torch.Size([2, 128])
```

---

## MultimodalSentimentModel

```python
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        #Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        #Classification
        self.emo_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,7) #Sadness , anger, 

        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,3) #Negative, positive, neutral

        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concantenate multimodal features
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1) # [batch_size, 128 * 3]


        # Fusion layer
        fused_features = self.fusion_layer(combined_features)


        emotion_output = self.emo_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotion': emotion_output,
            'sentiment': sentiment_output
        }
```

- `class MultimodalSentimentModel(nn.Module):` – Defines the top‑level multimodal model combining text, video, and audio.
- `def __init__(self):` – Constructor where encoders and classifiers are created.
- `super().__init__()` – Initializes the base `nn.Module`.
- Comment `#Encoders` – Indicates the section where encoders are instantiated.
- `self.text_encoder = TextEncoder()` – Creates a text encoder instance.
- `self.video_encoder = VideoEncoder()` – Creates a video encoder instance.
- `self.audio_encoder = AudioEncoder()` – Creates an audio encoder instance.
- Comment `#Fusion` – Marks the fusion block definition.
- `self.fusion_layer = nn.Sequential(... )` – Builds a small fusion network that combines the three 128‑dimensional modality features.
  - `nn.Linear(128 * 3, 256)` – Maps concatenated 384‑dimensional vector to 256 dimensions.
  - `nn.BatchNorm1d(256)` – Normalizes the 256 features across the batch.
  - `nn.ReLU()` – Non‑linear activation.
  - `nn.Dropout(0.3)` – Regularization with 30% dropout.
- Comment `#Classification` – Indicates the classifier heads.
- `self.emo_classifier = nn.Sequential(... )` – Defines the head for 7‑class emotion classification.
  - `nn.Linear(256, 128)` – First fully connected layer.
  - `nn.ReLU()` – Non‑linearity.
  - `nn.Dropout(0.2)` – Dropout regularization.
  - `nn.Linear(128,7)` – Final layer producing 7 logits (one per emotion class).
- `self.sentiment_classifier = nn.Sequential(... )` – Defines the head for 3‑class sentiment classification.
  - `nn.Linear(256, 128)` – First fully connected layer.
  - `nn.ReLU()` – Non‑linearity.
  - `nn.Dropout(0.2)` – Dropout.
  - `nn.Linear(128,3)` – Final layer producing 3 logits (negative, neutral, positive).
- `def forward(self, text_inputs, video_frames, audio_features):` – Forward pass definition.
- `text_features = self.text_encoder(... )` – Encodes the text inputs using the `TextEncoder`, producing `(B, 128)`.
- `video_features = self.video_encoder(video_frames)` – Encodes the video batch to `(B, 128)`.
- `audio_features = self.audio_encoder(audio_features)` – Encodes the audio batch to `(B, 128)`.
- Comment `# Concantenate multimodal features` – Explains that the next step concatenates features from all three modalities.
- `combined_features = torch.cat([...], dim=1)` – Concatenates the three `(B, 128)` tensors along the feature dimension into `(B, 384)`.
- Comment `# Fusion layer` – Indicates that the fusion network is applied next.
- `fused_features = self.fusion_layer(combined_features)` – Applies the fusion network to get a fused representation `(B, 256)`.
- `emotion_output = self.emo_classifier(fused_features)` – Produces emotion logits of shape `(B, 7)`.
- `sentiment_output = self.sentiment_classifier(fused_features)` – Produces sentiment logits of shape `(B, 3)`.
- `return { 'emotion': emotion_output, 'sentiment': sentiment_output }` – Returns a dictionary containing both outputs.

**Example usage:**

```python
model = MultimodalSentimentModel()

batch_size = 2
dummy_text = {
    'input_ids': torch.randint(0, 30522, (batch_size, 10)),
    'attention_mask': torch.ones(batch_size, 10, dtype=torch.long)
}
dummy_video = torch.randn(batch_size, 8, 3, 112, 112)
dummy_audio = torch.randn(batch_size, 1, 64, 100)

outputs = model(dummy_text, dummy_video, dummy_audio)
print(outputs['emotion'].shape)    # torch.Size([2, 7])
print(outputs['sentiment'].shape)  # torch.Size([2, 3])
```

---

## MultimodalTrainer

```python
class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        #Log dataset sized
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset Sizes: ")
        print(f"Train samples: {train_size} samples")
        print(f"Validation samples: {val_size} samples\n")
        print(f"Batch and epochs: {len(train_loader):,}")

        timestamp = datetime.now().strftime('%b%d_%H-%M-%S') #Feb12_20-44-00
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        # very high : 1, high: 0.1-0.01, medium: 1e-1, low 1e-4-le-5 very low: 1e-5
        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emo_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4} 
        ], weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode= 'min', 
            factor=0.1, 
            patience=2, 
        )

        self.current_train_losses = None

        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )
```

- `class MultimodalTrainer:` – Defines a helper class (not a `nn.Module`) that manages training and evaluation.
- `def __init__(self, model, train_loader, val_loader):` – Constructor taking a model and data loaders for training and validation.
- `self.model = model` – Stores the model instance.
- `self.train_loader = train_loader` – Stores the training data loader.
- `self.val_loader = val_loader` – Stores the validation data loader.
- Comment `#Log dataset sized` – Marks the section that prints dataset statistics.
- `train_size = len(train_loader.dataset)` – Gets the number of training samples.
- `val_size = len(val_loader.dataset)` – Gets the number of validation samples.
- `print("\nDataset Sizes: ")` – Prints a header with a leading newline.
- `print(f"Train samples: {train_size} samples")` – Prints the number of training samples.
- `print(f"Validation samples: {val_size} samples\n")` – Prints the number of validation samples, ending with a newline.
- `print(f"Batch and epochs: {len(train_loader):,}")` – Prints the number of batches per epoch, formatted with thousands separators.
- `timestamp = datetime.now().strftime('%b%d_%H-%M-%S')` – Generates a timestamp string (e.g., `Feb12_20-44-00`) for unique log directories.
- `base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'` – Chooses a base log directory: uses SageMaker's output path if running in that environment, otherwise local `runs`.
- `log_dir = f"{base_dir}/run_{timestamp}"` – Constructs the full log directory path.
- `self.writer = SummaryWriter(log_dir=log_dir)` – Creates a TensorBoard writer to log scalars and other summaries.
- `self.global_step = 0` – Initializes a global step counter used for logging over time.
- Comment about learning rates – Documents that different magnitudes of learning rates are used (very high to very low).
- `self.optimizer = torch.optim.Adam([...], weight_decay=1e-5)` – Creates an Adam optimizer with parameter groups and L2 weight decay.
  - Each dict in the list specifies a parameter group and its learning rate:
    - `model.text_encoder.parameters()` with `lr=8e-6` – Very low LR because the encoder is pretrained and mostly frozen.
    - `model.video_encoder.parameters()` and `model.audio_encoder.parameters()` with slightly higher LRs.
    - `model.fusion_layer`, `model.emo_classifier`, `model.sentiment_classifier` with `lr=5e-4` – Higher LRs for newly initialized layers.
- `self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)` – Creates a scheduler that reduces LR when a monitored metric (validation loss) stops improving.
  - `mode='min'` – Expects the metric to decrease.
  - `factor=0.1` – Multiplies LR by 0.1 when triggered.
  - `patience=2` – Waits for 2 epochs without improvement before reducing LR.
- `self.current_train_losses = None` – Placeholder to store the latest training losses, used when logging validation losses.
- `self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)` – Defines cross‑entropy loss with label smoothing for emotion prediction.
- `self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)` – Same, for sentiment prediction.

**Example usage (skeleton):**

```python
trainer = MultimodalTrainer(model, train_loader, val_loader)
for epoch in range(10):
    train_losses = trainer.train_epoch()
    val_losses, val_metrics = trainer.evaluate(val_loader, phase='Val')
```

---

### log_metrics

```python
    def log_metrics(self, losses, metrics=None, phase='Train'):
        if phase == 'Train':
            self.current_train_losses = losses
        else:
            self.writer.add_scalar(
                'loss/total/train', self.current_train_losses['total'], self.global_step)
            self.writer.add_scalar(
                'loss/total/val', losses['total'], self.global_step)
            
            self.writer.add_scalar(
                'loss/emotion/train', self.current_train_losses['emotion'], self.global_step)
            self.writer.add_scalar(
                'loss/emotion/val', losses['emotion'], self.global_step)

            self.writer.add_scalar(
                'loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step)
            self.writer.add_scalar(
                'loss/sentiment/val', losses['sentiment'], self.global_step)
        
        if metrics:
            self.writer.add_scalar(
                f"{phase}/emotion_precision", metrics['emotion_precision'], self.global_step
            )
            self.writer.add_scalar(
                f"{phase}/emotion_accuracy", metrics['emotion_accuracy'], self.global_step
            )
            self.writer.add_scalar(
                f"{phase}/sentiment_precision", metrics['sentiment_precision'], self.global_step
            )
            self.writer.add_scalar(
                f"{phase}/sentiment_accuracy", metrics['sentiment_accuracy'], self.global_step
            )
```

- `def log_metrics(self, losses, metrics=None, phase='Train'):` – Logs losses and optional metrics to TensorBoard; `phase` distinguishes Train vs Val/Test.
- `if phase == 'Train':` – If called during training steps:
  - `self.current_train_losses = losses` – Just store the latest training losses for later comparison/logging.
- `else:` – If called for validation or test:
  - `self.writer.add_scalar('loss/total/train', ...)` – Logs the previously stored training total loss.
  - `self.writer.add_scalar('loss/total/val', ...)` – Logs the current validation total loss.
  - Similar `add_scalar` calls log emotion and sentiment losses for both train and val.
- `if metrics:` – If a metrics dictionary is provided:
  - Logs precision and accuracy for both emotion and sentiment under keys like `Val/emotion_precision`.

**Example:**

```python
trainer.log_metrics(
    losses={'total': 1.0, 'emotion': 0.6, 'sentiment': 0.4},
    metrics={
        'emotion_precision': 0.75,
        'emotion_accuracy': 0.70,
        'sentiment_precision': 0.80,
        'sentiment_accuracy': 0.78,
    },
    phase='Val'
)
```

---

### train_epoch

```python
    def train_epoch(self):
        self.model.train()
        running_loss = { 'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}

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

            #Zero gradients
            self.optimizer.zero_grad()

            #Forward Pass
            outputs = self.model(text_inputs, video_frames, audio_features)

            #Calculate losses using raw logits
            emotion_loss = self.emotion_criterion(
                outputs['emotion'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(
                outputs['sentiment'], sentiment_labels)
            total_loss = emotion_loss + sentiment_loss


            #Backward and Calculate gradients
            total_loss.backward()

            #Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            self.optimizer.step()

            #Track losses
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

            self.log_metrics({
                'total': total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()

            })

            self.global_step += 1


        return {
            k: v/len(self.train_loader) for k,v in running_loss.items()
        }
```

- `def train_epoch(self):` – Runs one full epoch over the training data.
- `self.model.train()` – Sets the model to training mode (enables dropout, etc.).
- `running_loss = { ... }` – Initializes accumulators for total, emotion, and sentiment losses.
- `for batch in self.train_loader:` – Iterates over mini‑batches from the training loader.
- `device = next(self.model.parameters()).device` – Detects which device (CPU/GPU) the model is on.
- `text_inputs = {...}` – Prepares the text inputs dict, moving `input_ids` and `attention_mask` to the same device.
- `video_frames = batch['video_frames'].to(device)` – Moves video frames to device.
- `audio_features = batch['audio_features'].to(device)` – Moves audio features.
- `emotion_labels = batch['emotion_labels'].to(device)` – Moves emotion labels.
- `sentiment_labels = batch['sentiment_labels'].to(device)` – Moves sentiment labels.
- Comment `#Zero gradients` – Indicates gradient buffers are cleared.
- `self.optimizer.zero_grad()` – Resets gradients before the backward pass.
- Comment `#Forward Pass` – Next lines perform inference.
- `outputs = self.model(text_inputs, video_frames, audio_features)` – Runs the model, obtaining emotion and sentiment logits.
- Comment `#Calculate losses using raw logits` – The next few lines compute the losses.
- `emotion_loss = self.emotion_criterion(outputs['emotion'], emotion_labels)` – Computes cross‑entropy loss for emotion.
- `sentiment_loss = self.sentiment_criterion(outputs['sentiment'], sentiment_labels)` – Computes cross‑entropy loss for sentiment.
- `total_loss = emotion_loss + sentiment_loss` – Adds the two losses to form a joint objective.
- Comment `#Backward and Calculate gradients` – Start backprop.
- `total_loss.backward()` – Backpropagates to compute gradients.
- Comment `#Gradient Clipping` – Explains the next step limits gradient size.
- `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)` – Clips gradients to prevent exploding gradients.
- `self.optimizer.step()` – Updates model parameters using the computed gradients.
- Comment `#Track losses` – Now accumulation of losses.
- `running_loss[...] += ...` – Accumulates total, emotion, and sentiment loss values for epoch‑level averages.
- `self.log_metrics({...})` – Logs the current batch losses via `log_metrics` (with default `phase='Train'`).
- `self.global_step += 1` – Increments the global step counter.
- `return { k: v/len(self.train_loader) ... }` – Returns average losses across all batches.

---

### evaluate

```python
    def evaluate(self, data_loader, phase='Val'):
        self.model.eval()
        losses = { 'total': 0, 'emotion': 0 , 'sentiment': 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

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

                # Calculate losses using raw logits
                emotion_loss = self.emotion_criterion(
                    outputs['emotion'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(
                    outputs['sentiment'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss

                all_emotion_preds.extends(
                    outputs["emotion"].argmax(dim=1).cpu().numpy())

                all_emotion_labels.extend(
                    emotion_labels.cpu().numpy())

                all_sentiment_preds.extend(
                    outputs["sentiment"].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(
                    sentiment_labels.cpu().numpy())


                #Tack losses
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()

        
        avg_loss = { k : v/len(data_loader) for k,v in losses.items()}
        

        # Copute precision and accuracy
        emotion_precision = precision_score(
            all_emotion_labels, all_emotion_preds, average='weighted')
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
        sentiment_precision = precision_score(
            all_sentiment_labels, all_sentiment_preds, average='weighted')
        sentiment_accuracy = accuracy_score(
            all_sentiment_labels, all_sentiment_preds)
        
        self.log_metrics(avg_loss,{
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

- `def evaluate(self, data_loader, phase='Val'):` – Runs evaluation on the provided data loader; `phase` label is used when logging.
- `self.model.eval()` – Puts the model in evaluation mode (disables dropout, uses running batchnorm stats).
- `losses = {...}` – Initializes loss accumulators.
- `all_emotion_preds`, `all_emotion_labels`, `all_sentiment_preds`, `all_sentiment_labels` – Lists to collect predictions and true labels for metrics.
- `with torch.inference_mode():` – Disables gradient calculation for efficiency and memory savings.
- `for batch in data_loader:` – Iterates over evaluation batches.
- Inside the loop, the preparation of `text_inputs`, `video_frames`, `audio_features`, `emotion_labels`, `sentiment_labels` is analogous to `train_epoch`, moving everything to the correct device.
- Comment `# Forward Pass` – Indicates model inference.
- `outputs = self.model(...)` – Runs the model to get logits.
- Comment `# Calculate losses using raw logits` – Next lines compute losses per batch.
- `emotion_loss`, `sentiment_loss`, `total_loss` – Same structure as in training, but without backward.
- `all_emotion_preds.extends(...)` – Intends to append predicted emotion class indices; note there is a small typo here: it should be `extend`, not `extends`.
- `all_emotion_labels.extend(...)` – Appends true emotion labels.
- `all_sentiment_preds.extend(...)` – Appends predicted sentiment class indices.
- `all_sentiment_labels.extend(...)` – Appends true sentiment labels.
- Comment `#Tack losses` – Should likely read `#Track losses`; it accumulates batch losses.
- `losses[...] += ...` – Adds batch losses for averaging.
- `avg_loss = { k : v/len(data_loader) ... }` – Computes average losses.
- Comment `# Copute precision and accuracy` – Computes evaluation metrics using scikit‑learn.
- `emotion_precision = precision_score(...)` – Weighted precision for emotion.
- `emotion_accuracy = accuracy_score(...)` – Accuracy for emotion.
- Similarly for `sentiment_precision` and `sentiment_accuracy`.
- `self.log_metrics(avg_loss, {...}, phase=phase)` – Logs losses and metrics to TensorBoard.
- `if phase == 'Val': self.scheduler.step(avg_loss['total'])` – Adjusts learning rate based on validation total loss.
- `return avg_loss, {...}` – Returns average losses and metric dictionary.

**Example (validation):**

```python
val_losses, val_metrics = trainer.evaluate(val_loader, phase='Val')
print(val_losses)
print(val_metrics)
```

---

## __main__ Example Block

```python
if __name__ == "__main__":
    dataset = MELDDataset(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits')

    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }

    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)


    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)

        emotions_probs = torch.softmax(outputs['emotion'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiment'], dim=1)[0]

    emotion_map ={
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'
        # 'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6
    }

    sentiment_map = {
        0: 'negative', 1: 'neutral', 2: 'positive'
        # 'negative': 0, 'neutral': 1, 'positive': 2 
    }

    for i , prob in enumerate(emotions_probs):
        print(f"{emotion_map[i]}: {prob:.2f}")
    
    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")

    print("Emotion Probabilities:", emotions_probs)
```

- `if __name__ == "__main__":` – Ensures that the following example code runs only when this file is executed directly, not when imported.
- `dataset = MELDDataset(...)` – Creates a MELD training dataset instance using the CSV file and split directory.
- `sample = dataset[0]` – Retrieves the first sample from the dataset.
- `model = MultimodalSentimentModel()` – Instantiates the multimodal model.
- `model.eval()` – Puts the model in evaluation mode.
- `text_inputs = {...}` – Constructs a mini‑batch of size 1 for text by adding a batch dimension with `unsqueeze(0)`.
- `video_frames = sample['video_frames'].unsqueeze(0)` – Adds batch dimension to video data.
- `audio_features = sample['audio_features'].unsqueeze(0)` – Adds batch dimension to audio data.
- `with torch.inference_mode():` – Disables gradient computation for inference.
- `outputs = model(text_inputs, video_frames, audio_features)` – Runs the model to get logits for this single sample.
- `emotions_probs = torch.softmax(outputs['emotion'], dim=1)[0]` – Applies softmax across the emotion logits to obtain probabilities and selects the first (only) batch element.
- `sentiment_probs = torch.softmax(outputs['sentiment'], dim=1)[0]` – Same for sentiment.
- `emotion_map = { ... }` – Maps each emotion class index to a human‑readable string.
- Comment underneath shows the inverse mapping for clarity.
- `sentiment_map = { ... }` – Maps sentiment indices to labels.
- Comment underneath again shows the inverse mapping.
- `for i , prob in enumerate(emotions_probs): print(...)` – Iterates through emotion probabilities and prints a label with its probability (rounded to 2 decimal places).
- `for i, prob in enumerate(sentiment_probs): print(...)` – Does the same for sentiment probabilities.
- `print("Emotion Probabilities:", emotions_probs)` – Prints the raw tensor of emotion probabilities.

**How to run this example:**

From your project root (ensuring paths are correct):

```bash
python training/models.py
```

This will print out the predicted emotion and sentiment probabilities for the first sample in the MELD training set.

---

This completes the line‑by‑line explanation of `models.py` with practical examples for how to use each component.
