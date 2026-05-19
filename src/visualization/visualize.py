from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_training_history(
    history: dict[str, list[float]],
    save_path: Path | None = None,
) -> None:
    has_gap = bool(history.get("acc_gap"))
    ncols = 3 if has_gap else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 2:
        axes = list(axes)

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    if has_gap:
        axes[2].plot(epochs, history["acc_gap"], color="crimson")
        axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[2].set_title("Overfit gap (train - val acc)")
        axes[2].set_xlabel("Epoch")

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
