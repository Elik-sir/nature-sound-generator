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
        self._lr = lr
        self._weight_decay = weight_decay
        self._reset_optimizer()

    def _reset_optimizer(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            params, lr=self._lr, weight_decay=self._weight_decay
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
        early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE,
        label_smoothing: float = config.LABEL_SMOOTHING,
        mixup_alpha: float = config.MIXUP_ALPHA,
        overfit_acc_gap: float = config.OVERFIT_ACC_GAP,
        overfit_gap_patience: int = config.OVERFIT_GAP_PATIENCE,
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
            "acc_gap": [],
        }
        best_val_loss = float("inf")
        best_state: dict | None = None
        val_loss_patience = 0
        gap_patience = 0
        stopped_early = False
        stop_reason = ""

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(
                train_loader, criterion, train=True, mixup_alpha=mixup_alpha
            )
            val_loss, val_acc = self._run_epoch(val_loader, criterion, train=False)
            scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            acc_gap = train_acc - val_acc

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["acc_gap"].append(acc_gap)

            print(
                f"Epoch {epoch}/{epochs} lr={current_lr:.2e} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"gap={acc_gap:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                val_loss_patience = 0
                if save_checkpoints:
                    checkpoints.save_best(
                        self.model_name,
                        self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics={
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "acc_gap": acc_gap,
                        },
                        history=history,
                        extra={
                            "epochs_planned": epochs,
                            "cnn_arch": config.CNN_ARCH,
                            "lr": self._lr,
                            "weight_decay": self._weight_decay,
                        },
                    )
            else:
                val_loss_patience += 1

            if acc_gap > overfit_acc_gap:
                gap_patience += 1
            else:
                gap_patience = 0

            if gap_patience >= overfit_gap_patience:
                stopped_early = True
                stop_reason = (
                    f"overfitting (acc gap > {overfit_acc_gap:.2f} for "
                    f"{overfit_gap_patience} epochs)"
                )
                print(f"Early stopping: {stop_reason}")
                break

            if val_loss_patience >= early_stopping_patience:
                stopped_early = True
                stop_reason = (
                    f"no val_loss improvement for {early_stopping_patience} epochs"
                )
                print(f"Early stopping: {stop_reason} (best val_loss={best_val_loss:.4f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(
                f"Restored best weights (val_loss={best_val_loss:.4f}). "
                "Use these for test evaluation, not the last epoch."
            )

        if stopped_early:
            history.setdefault("notes", []).append(stop_reason)
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

    def _run_lstm_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        train: bool,
    ) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        correct = 0.0
        total = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device).float()

            if train:
                self.optimizer.zero_grad()

            logits = self.model(features).squeeze(-1)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                self.optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).float().sum().item()
            total += labels.numel()
            total_loss += loss.item() * labels.numel()

        return total_loss / max(total, 1), correct / max(total, 1)

    def train_lstm_detector(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = config.EPOCHS_LSTM,
        *,
        save_checkpoints: bool = True,
    ) -> dict[str, list[float]]:
        criterion = nn.BCEWithLogitsLoss()
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        best_val_loss = float("inf")
        best_state: dict | None = None

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_lstm_epoch(train_loader, criterion, train=True)
            val_loss, val_acc = self._run_lstm_epoch(val_loader, criterion, train=False)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                if save_checkpoints:
                    checkpoints.save_best(
                        self.model_name,
                        self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics={
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                        },
                        history=history,
                        extra={
                            "epochs_planned": epochs,
                            "lr": self._lr,
                            "weight_decay": self._weight_decay,
                        },
                    )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(
                f"Restored best LSTM weights (val_loss={best_val_loss:.4f}). "
                "Use these for test evaluation."
            )
        return history

    def predict_lstm_scores(self, loader: DataLoader) -> tuple[list[float], list[float]]:
        self.model.eval()
        y_true: list[float] = []
        y_score: list[float] = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                logits = self.model(features).squeeze(-1)
                probs = torch.sigmoid(logits).cpu()
                y_score.extend(probs.flatten().tolist())
                y_true.extend(labels.flatten().tolist())

        return y_true, y_score

    def _run_vae_epoch(
        self,
        loader: DataLoader,
        train: bool,
        *,
        beta_kl: float = 1.0,
        delta_weight: float = config.VAE_DELTA_LOSS_WEIGHT,
    ) -> tuple[float, float, float]:
        self.model.train(train)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total = 0

        for features, _ in loader:
            features = features.to(self.device)
            if train:
                self.optimizer.zero_grad()

            recon, mu, logvar = self.model(features)
            recon_l1 = nn.functional.l1_loss(recon, features, reduction="mean")
            time_delta_recon = recon[:, :, :, 1:] - recon[:, :, :, :-1]
            time_delta_target = features[:, :, :, 1:] - features[:, :, :, :-1]
            freq_delta_recon = recon[:, :, 1:, :] - recon[:, :, :-1, :]
            freq_delta_target = features[:, :, 1:, :] - features[:, :, :-1, :]
            time_delta_loss = nn.functional.l1_loss(
                time_delta_recon, time_delta_target, reduction="mean"
            )
            freq_delta_loss = nn.functional.l1_loss(
                freq_delta_recon, freq_delta_target, reduction="mean"
            )
            recon_loss = recon_l1 + delta_weight * (time_delta_loss + freq_delta_loss)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta_kl * kl_loss

            if train:
                loss.backward()
                self.optimizer.step()

            batch_size = features.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size

        denom = max(total, 1)
        return total_loss / denom, total_recon / denom, total_kl / denom

    def train_vae(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = config.EPOCHS_VAE,
        *,
        beta_kl: float = config.VAE_BETA_KL,
        kl_warmup_epochs: int = config.VAE_KL_WARMUP_EPOCHS,
        delta_weight: float = config.VAE_DELTA_LOSS_WEIGHT,
        save_checkpoints: bool = True,
    ) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_recon": [],
            "train_kl": [],
            "val_loss": [],
            "val_recon": [],
            "val_kl": [],
            "beta_kl": [],
        }
        best_val_loss = float("inf")
        best_state: dict | None = None

        for epoch in range(1, epochs + 1):
            warmup_ratio = min(1.0, epoch / max(1, kl_warmup_epochs))
            current_beta = beta_kl * warmup_ratio
            train_loss, train_recon, train_kl = self._run_vae_epoch(
                train_loader, train=True, beta_kl=current_beta, delta_weight=delta_weight
            )
            val_loss, val_recon, val_kl = self._run_vae_epoch(
                val_loader, train=False, beta_kl=current_beta, delta_weight=delta_weight
            )

            history["train_loss"].append(train_loss)
            history["train_recon"].append(train_recon)
            history["train_kl"].append(train_kl)
            history["val_loss"].append(val_loss)
            history["val_recon"].append(val_recon)
            history["val_kl"].append(val_kl)
            history["beta_kl"].append(current_beta)

            print(
                f"Epoch {epoch}/{epochs} "
                f"beta_kl={current_beta:.3f} "
                f"train_loss={train_loss:.4f} train_recon={train_recon:.4f} train_kl={train_kl:.4f} "
                f"val_loss={val_loss:.4f} val_recon={val_recon:.4f} val_kl={val_kl:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                if save_checkpoints:
                    checkpoints.save_best(
                        self.model_name,
                        self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics={
                            "train_loss": train_loss,
                            "train_recon": train_recon,
                            "train_kl": train_kl,
                            "val_loss": val_loss,
                            "val_recon": val_recon,
                            "val_kl": val_kl,
                        },
                        history=history,
                        extra={
                            "epochs_planned": epochs,
                            "beta_kl": beta_kl,
                            "kl_warmup_epochs": kl_warmup_epochs,
                            "delta_weight": delta_weight,
                            "lr": self._lr,
                            "weight_decay": self._weight_decay,
                        },
                    )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(
                f"Restored best VAE weights (val_loss={best_val_loss:.4f}). "
                "Use these for reconstruction evaluation."
            )
        return history
