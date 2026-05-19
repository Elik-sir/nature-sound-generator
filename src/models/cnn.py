from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src import config


class LightCNNClassifier(nn.Module):
    """Compact CNN for small datasets (~300k params). Harder to overfit than ResNet-18."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.CNN_DROPOUT,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class ResNet18Classifier(nn.Module):
    """Optional larger backbone — use only if you accept higher overfitting risk."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.CNN_DROPOUT,
        pretrained: bool = config.CNN_PRETRAINED,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if pretrained:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                backbone.conv1.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )
        else:
            backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self, train_last_block: bool = False) -> None:
        for name, param in self.backbone.named_parameters():
            if name.startswith("fc"):
                param.requires_grad = True
            elif train_last_block and name.startswith("layer4"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# Default export — light model for ESC-50 scale
CNNClassifier = LightCNNClassifier


def build_cnn_classifier(arch: str | None = None) -> nn.Module:
    arch = (arch or config.CNN_ARCH).lower()
    if arch in ("light", "lightcnn"):
        return LightCNNClassifier()
    if arch == "resnet18":
        return ResNet18Classifier()
    raise ValueError(f"Unknown CNN_ARCH: {arch!r}. Use 'light' or 'resnet18'.")
