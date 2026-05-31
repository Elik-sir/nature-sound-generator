from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src import config


class ConcatPool2d(nn.Module):
    """
    Для аудио: конкатенирует Max (ловит резкие звуки, напр. птиц/хруст) 
    и Avg (ловит фоновые звуки, напр. ветер/дождь) пулинг.
    """
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.mp = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.ap(x), self.mp(x)], dim=1)


class LightCNNClassifier(nn.Module):
    """Compact CNN optimized for audio spectrograms."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.CNN_DROPOUT,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Блок 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Блок 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Блок 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.pool = ConcatPool2d()
        
        # Увеличиваем in_features в 2 раза из-за ConcatPool2d
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResNet18Classifier(nn.Module):
    """ResNet-18 adapted for audio with 1-channel input and ConcatPooling."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.CNN_DROPOUT,
        pretrained: bool = config.CNN_PRETRAINED,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Адаптация под 1 канал
        if pretrained:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                # Усреднение весов 3-х каналов в 1 для сохранения претрейна
                backbone.conv1.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )
        else:
            backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Заменяем стандартный пулинг на аудио-ориентированный
        backbone.avgpool = ConcatPool2d()

        # In_features * 2 из-за ConcatPool2d
        in_features = backbone.fc.in_features * 2
        
        backbone.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
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


# Default export
CNNClassifier = LightCNNClassifier


def build_cnn_classifier(arch: str | None = None) -> nn.Module:
    arch = (arch or config.CNN_ARCH).lower()
    if arch in ("light", "lightcnn"):
        return LightCNNClassifier()
    if arch == "resnet18":
        return ResNet18Classifier()
    raise ValueError(f"Unknown CNN_ARCH: {arch!r}. Use 'light' or 'resnet18'.")