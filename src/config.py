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

BATCH_SIZE_CNN = 32
BATCH_SIZE_LSTM = 8
BATCH_SIZE_VAE = 8
EPOCHS_CNN = 80
EPOCHS_LSTM = 5
EPOCHS_VAE = 50
VAE_TARGET_CLASS = TARGET_CLASS
VAE_KL_WARMUP_EPOCHS = 10
VAE_CROP_SECONDS = 4.0
VAE_BETA_KL = 0.05
VAE_LOG_MEL_MIN = -12.0
VAE_LOG_MEL_MAX = 2.0
VAE_DELTA_LOSS_WEIGHT = 0.5

# CNN: "light" (~300k params) generalizes better on ESC-50; "resnet18" overfits easily
CNN_ARCH = "light"
CNN_PRETRAINED = False

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
CNN_DROPOUT = 0.5
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 8
MIXUP_ALPHA = 0.3

# Stop when train_acc - val_acc stays above this gap (clear overfitting)
OVERFIT_ACC_GAP = 0.12
OVERFIT_GAP_PATIENCE = 4

LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

# Mel augmentation (train only) — stronger to narrow train/val gap
AUG_FREQ_MASK_PROB = 0.9
AUG_TIME_MASK_PROB = 0.9
AUG_FREQ_MASK_MAX = 24
AUG_TIME_MASK_MAX = 24
AUG_NOISE_PROB = 0.4
AUG_NOISE_STD = 0.08
AUG_GAIN_MIN = 0.85
AUG_GAIN_MAX = 1.15

# Audio waveform augmentation (applied before mel conversion, train only)
ENABLE_AUDIO_AUGMENT = True
AUDIO_AUG_PROB = 0.5
AUDIO_SHIFT_MAX_SECONDS = 0.2
AUDIO_NOISE_STD = 0.01
AUDIO_GAIN_MIN = 0.8
AUDIO_GAIN_MAX = 1.2


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
