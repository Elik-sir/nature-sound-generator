from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src import config


class CNNClassifier(nn.Module):
    """ResNet-18 backbone for 128×128 mel spectrograms (1 channel)."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.CNN_DROPOUT,
    ):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
