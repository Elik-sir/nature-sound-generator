from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.models import checkpoints


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        lr: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        require_cuda: bool = False,
        model_name: str = "cnn",
    ):
        self.device = device or config.get_device(require_cuda=require_cuda)
        self.model_name = model_name
        print(f"Training on: {config.describe_device(self.device)}")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    @staticmethod
    def _mixup_batch(
        features: torch.Tensor,
        labels: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = float(np.random.beta(alpha, alpha))
        index = torch.randperm(features.size(0), device=features.device)
        mixed = lam * features + (1.0 - lam) * features[index]
        return mixed, labels, labels[index], lam

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        train: bool,
        *,
        mixup_alpha: float = 0.0,
    ) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            if train and mixup_alpha > 0:
                features, labels_a, labels_b, lam = self._mixup_batch(
                    features, labels, mixup_alpha
                )
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            if train:
                self.optimizer.zero_grad()

            logits = self.model(features)
            if train and mixup_alpha > 0:
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(
                    logits, labels_b
                )
            else:
                loss = criterion(logits, labels)

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            if train and mixup_alpha > 0:
                correct += (
                    lam * (preds == labels_a).float()
                    + (1.0 - lam) * (preds == labels_b).float()
                ).sum().item()
            else:
                correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    def train_classifier(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = config.EPOCHS_CNN,
        *,
        save_checkpoints: bool = True,
        checkpoint_every: int = 1,
        early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE,
        label_smoothing: float = config.LABEL_SMOOTHING,
        mixup_alpha: float = config.MIXUP_ALPHA,
    ) -> dict[str, list[float]]:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
        )
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0
        last_checkpoint: checkpoints.CheckpointInfo | None = None
        stopped_early = False

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(
                train_loader, criterion, train=True, mixup_alpha=mixup_alpha
            )
            val_loss, val_acc = self._run_epoch(val_loader, criterion, train=False)
            scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch}/{epochs} lr={current_lr:.2e} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if save_checkpoints and epoch % checkpoint_every == 0:
                metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                extra = {
                    "epochs_planned": epochs,
                    "lr": config.LEARNING_RATE,
                    "weight_decay": config.WEIGHT_DECAY,
                    "label_smoothing": label_smoothing,
                }
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                    last_checkpoint = checkpoints.save_checkpoint(
                        self.model_name,
                        self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=metrics,
                        history=history,
                        extra=extra,
                        is_best=True,
                    )
                else:
                    checkpoints.save_latest_snapshot(
                        self.model_name,
                        self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=metrics,
                        history=history,
                        extra=extra,
                    )

            if patience_counter >= early_stopping_patience:
                print(
                    f"Early stopping: no val_loss improvement for "
                    f"{early_stopping_patience} epochs (best val_loss={best_val_loss:.4f})"
                )
                stopped_early = True
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored weights with best val_loss={best_val_loss:.4f}")

        if save_checkpoints and last_checkpoint is not None:
            best = checkpoints.get_best_checkpoint(self.model_name)
            if best:
                print(
                    f"Best checkpoint (by val_acc): v{best.version:03d} "
                    f"val_acc={best.val_acc:.4f} ({best.path})"
                )

        if stopped_early:
            history.setdefault("notes", []).append("early_stopping")
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
