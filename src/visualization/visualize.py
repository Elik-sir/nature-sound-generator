from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from src import config


def plot_training_history(
    history: dict[str, list[float]],
    save_path: Path | None = None,
) -> None:
    has_acc = "train_acc" in history and "val_acc" in history
    has_gap = bool(history.get("acc_gap"))
    has_vae_parts = "train_recon" in history and "train_kl" in history

    ncols = 1
    if has_acc:
        ncols += 1
    if has_gap:
        ncols += 1
    if has_vae_parts:
        ncols += 2

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if not isinstance(axes, (list, tuple)):
        axes = [axes] if ncols == 1 else list(axes)

    epochs = range(1, len(history["train_loss"]) + 1)
    idx = 0
    axes[idx].plot(epochs, history["train_loss"], label="train")
    axes[idx].plot(epochs, history["val_loss"], label="val")
    axes[idx].set_title("Loss")
    axes[idx].set_xlabel("Epoch")
    axes[idx].legend()
    idx += 1

    if has_acc:
        axes[idx].plot(epochs, history["train_acc"], label="train")
        axes[idx].plot(epochs, history["val_acc"], label="val")
        axes[idx].set_title("Accuracy")
        axes[idx].set_xlabel("Epoch")
        axes[idx].legend()
        idx += 1

    if has_gap:
        axes[idx].plot(epochs, history["acc_gap"], color="crimson")
        axes[idx].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[idx].set_title("Overfit gap (train - val acc)")
        axes[idx].set_xlabel("Epoch")
        idx += 1

    if has_vae_parts:
        axes[idx].plot(epochs, history["train_recon"], label="train")
        axes[idx].plot(epochs, history["val_recon"], label="val")
        axes[idx].set_title("Reconstruction Loss")
        axes[idx].set_xlabel("Epoch")
        axes[idx].legend()
        idx += 1

        axes[idx].plot(epochs, history["train_kl"], label="train")
        axes[idx].plot(epochs, history["val_kl"], label="val")
        axes[idx].set_title("KL Divergence")
        axes[idx].set_xlabel("Epoch")
        axes[idx].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] | None = None,
    save_path: Path | None = None,
) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    display_labels = class_names if class_names else labels

    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return overall and per-class metrics for multiclass classification."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
    }

    per_class: dict[str, dict[str, float]] = {}
    for label, values in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        per_class[label] = {
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "f1": float(values["f1-score"]),
            "support": float(values["support"]),
        }
    return overall, per_class


def print_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> None:
    overall, per_class = classification_metrics(y_true, y_pred)
    print("Overall metrics:")
    print(
        f"  Accuracy={overall['accuracy']:.4f} | "
        f"Precision(macro)={overall['precision_macro']:.4f} | "
        f"Recall(macro)={overall['recall_macro']:.4f} | "
        f"F1(macro)={overall['f1_macro']:.4f}"
    )
    print(
        f"  Precision(weighted)={overall['precision_weighted']:.4f} | "
        f"Recall(weighted)={overall['recall_weighted']:.4f} | "
        f"F1(weighted)={overall['f1_weighted']:.4f}"
    )

    print("\nPer-class metrics (precision / recall / f1 / support):")
    for label in sorted(per_class, key=lambda x: int(x) if x.isdigit() else x):
        m = per_class[label]
        print(
            f"  class {label}: p={m['precision']:.3f}, r={m['recall']:.3f}, "
            f"f1={m['f1']:.3f}, n={int(m['support'])}"
        )


def plot_roc_curve(
    y_true: Sequence[float],
    y_score: Sequence[float],
    save_path: Path | None = None,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return roc_auc


def plot_spectrogram_pair(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: Path | None = None,
) -> None:
    orig = original.detach().cpu().squeeze()
    recon = reconstructed.detach().cpu().squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(orig, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Original")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mel bins")

    axes[1].imshow(recon, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Reconstruction")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Mel bins")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def mel_to_audio(
    mel: torch.Tensor,
    *,
    sample_rate: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    n_mels: int = config.MEL_N_MELS,
    n_iter: int = 32,
    input_scale: str = "power",
) -> torch.Tensor:
    mel_tensor = mel.detach().cpu().float()
    if mel_tensor.dim() == 2:
        mel_tensor = mel_tensor.unsqueeze(0)
    if mel_tensor.dim() == 4:
        mel_tensor = mel_tensor.squeeze(0)

    if mel_tensor.shape[-2:] != (n_mels, config.MEL_SIZE):
        mel_tensor = F.interpolate(
            mel_tensor.unsqueeze(0),
            size=(n_mels, config.MEL_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if input_scale == "log":
        mel_tensor = mel_tensor.clamp(
            min=config.VAE_LOG_MEL_MIN, max=config.VAE_LOG_MEL_MAX
        )
        mel_power = torch.exp(mel_tensor).clamp_min(1e-6)
    elif input_scale == "power":
        mel_power = torch.relu(mel_tensor) + 1e-6
    else:
        raise ValueError(f"Unsupported input_scale: {input_scale!r}. Use 'power' or 'log'.")
    inverse_mel = T.InverseMelScale(
        n_stft=(n_fft // 2) + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )
    stft_mag = inverse_mel(mel_power)
    griffin = T.GriffinLim(n_fft=n_fft, hop_length=hop_length, n_iter=n_iter, power=1.0)
    waveform = griffin(stft_mag)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform
