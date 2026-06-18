"""Model architectures extracted from the Master's project notebooks.

Used to load the trained .pth weights and export them to ONNX. The backend
serves the ONNX models, so torch is only needed at conversion time.

  - PieceImageClassifier : MobileNetV2 -> 12 classes (square -> piece, digitization)
  - PieceClassifier      : CNN, 12x8x8 -> 6  (which piece type to move)
  - SquareClassifier     : CNN, 12x8x8 -> 64 (destination square; one per piece type)
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


class PieceImageClassifier(nn.Module):
    """Digitization: classify a single board square crop into one of 12 pieces."""

    def __init__(self):
        super().__init__()
        # weights=None: we load our own trained state_dict, no ImageNet download.
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 12),
        )

    def forward(self, x):
        return self.model.classifier(self.model.features(x))


def _conv_block() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(12, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )


class PieceClassifier(nn.Module):
    """Move prediction stage 1: which piece type (P,N,B,R,Q,K) is likely to move."""

    def __init__(self):
        super().__init__()
        self.conv_layers = _conv_block()
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 6),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class SquareClassifier(nn.Module):
    """Move prediction stage 2: destination square (0-63) for a given piece type."""

    def __init__(self):
        super().__init__()
        self.conv_layers = _conv_block()
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 64),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
