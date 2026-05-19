from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = PROJECT_ROOT / "ESC-50-master" / "audio"
META_CSV = PROJECT_ROOT / "ESC-50-master" / "meta" / "esc50.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

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
EPOCHS_CNN = 20
EPOCHS_LSTM = 30
EPOCHS_VAE = 50
LEARNING_RATE = 1e-3


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_num_workers() -> int:
    return 0 if sys.platform == "win32" else 2


def require_dataset() -> None:
    if not RAW_AUDIO_DIR.is_dir() or not any(RAW_AUDIO_DIR.glob("*.wav")):
        raise FileNotFoundError(
            "ESC-50 audio not found. Download from "
            "https://github.com/karolpiczak/ESC-50 and extract to "
            f"{PROJECT_ROOT / 'ESC-50-master'}/"
        )
