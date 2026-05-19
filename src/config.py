from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = PROJECT_ROOT / "ESC-50-master" / "audio"
META_CSV = PROJECT_ROOT / "ESC-50-master" / "meta" / "esc50.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"

TARGET_CLASS = 14
LSTM_DURATION_S = 30
TEST_FOLD = 5
VAL_FOLD = 4
NUM_CLASSES = 50
MEL_N_MELS = 128
MEL_SIZE = 128
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
RANDOM_SEED = 42

BATCH_SIZE_CNN = 64
BATCH_SIZE_LSTM = 8
EPOCHS_CNN = 100
EPOCHS_LSTM = 30
EPOCHS_VAE = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CNN_DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 12
MIXUP_ALPHA = 0.2
LR_SCHEDULER_PATIENCE = 4
LR_SCHEDULER_FACTOR = 0.5

# Mel augmentation (train only)
AUG_FREQ_MASK_PROB = 0.5
AUG_TIME_MASK_PROB = 0.5
AUG_FREQ_MASK_MAX = 16
AUG_TIME_MASK_MAX = 16
AUG_NOISE_PROB = 0.3
AUG_NOISE_STD = 0.05


def get_device(*, require_cuda: bool = False) -> torch.device:
    """Return training device. Set FORCE_CPU=1 to override. Set require_cuda=True to fail without GPU."""
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        if require_cuda:
            raise RuntimeError("FORCE_CPU is set but CUDA was required.")
        return torch.device("cpu")

    if torch.cuda.is_available():
        index = int(os.environ.get("CUDA_DEVICE", "0"))
        return torch.device(f"cuda:{index}")

    if require_cuda:
        raise RuntimeError(
            "CUDA is not available. Reinstall PyTorch with GPU support, e.g.\n"
            "  uv sync\n"
            "after configuring the pytorch-cu126 index in pyproject.toml, then verify:\n"
            "  uv run python -c \"import torch; print(torch.cuda.is_available())\""
        )
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> str:
    device = device or get_device()
    if device.type != "cuda":
        return f"cpu (torch {torch.__version__})"
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    return f"cuda:{index} ({name}, torch {torch.__version__})"


def get_num_workers() -> int:
    return 0 if sys.platform == "win32" else 2


def require_dataset() -> None:
    if not RAW_AUDIO_DIR.is_dir() or not any(RAW_AUDIO_DIR.glob("*.wav")):
        raise FileNotFoundError(
            "ESC-50 audio not found. Download from "
            "https://github.com/karolpiczak/ESC-50 and extract to "
            f"{PROJECT_ROOT / 'ESC-50-master'}/"
        )
