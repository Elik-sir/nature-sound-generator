from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        lr: float = config.LEARNING_RATE,
    ):
        self.device = device or config.get_device()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        train: bool,
    ) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            if train:
                self.optimizer.zero_grad()

            logits = self.model(features)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    def train_classifier(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = config.EPOCHS_CNN,
    ) -> dict[str, list[float]]:
        criterion = nn.CrossEntropyLoss()
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(
                train_loader, criterion, train=True
            )
            val_loss, val_acc = self._run_epoch(val_loader, criterion, train=False)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        return history

    def predict_classifier(self, loader: DataLoader) -> tuple[list[int], list[int]]:
        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                logits = self.model(features)
                preds = logits.argmax(dim=1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.tolist())

        return y_true, y_pred
