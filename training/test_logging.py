import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import MultimodalTrainer

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Linear(1, 1)
        self.video_encoder = nn.Linear(1, 1)
        self.audio_encoder = nn.Linear(1, 1)
        self.fusion_layer = nn.Linear(3, 1)
        self.emo_classifier = nn.Linear(1, 7)
        self.sentiment_classifier = nn.Linear(1, 3)

def test_logging():
    samples = [
        {"emotion_labels": i, "sentiment_labels": i % 3}
        for i in range(7)
    ]
    mock_loader = DataLoader(samples, batch_size=2)
    trainer = MultimodalTrainer(MockModel(), mock_loader, mock_loader)

    trainer.log_metrics({"total": 2.5, "emotion": 1.0, "sentiment": 1.5}, phase="Train")
    trainer.log_metrics(
        {"total": 1.5, "emotion": 0.5, "sentiment": 1.0},
        {"emotion_precision": 0.65, "emotion_accuracy": 0.75, "sentiment_precision": 0.85, "sentiment_accuracy": 0.95},
        phase="Val",
    )

if __name__ == "__main__":
    test_logging()