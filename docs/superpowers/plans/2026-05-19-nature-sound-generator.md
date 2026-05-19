# Nature Sound Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CCDS Lite PyTorch lab on ESC-50 with CNN classification, LSTM frame-level detection on synthetic 30 s tracks, and VAE mel synthesis.

**Architecture:** Hybrid data pipeline — preprocess 5 s clips to `data/processed/*.pt`; CNN/VAE read cached mels; LSTM stitches waveforms at runtime with per-frame labels. Fold-based splits, `uv` for deps, no audio in git.

**Tech Stack:** Python 3.11+, PyTorch, torchaudio, uv, scikit-learn, matplotlib, seaborn, pytest

**Design spec:** `docs/superpowers/specs/2026-05-19-nature-sound-generator-design.md`

**Gate:** Stop after each **Phase** and wait for user approval before starting the next.

---

## File map

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Project metadata, deps, `[tool.pytest]` |
| `src/config.py` | Paths, mel params, folds, device helper |
| `src/data/make_dataset.py` | wav → mel 128×128 `.pt` + manifest |
| `src/data/dataloaders.py` | `ESC50Dataset`, `LSTMSyntheticDataset`, `get_dataloaders` |
| `src/models/cnn.py` | `CNNClassifier` |
| `src/models/lstm.py` | `AudioLSTMDetector` (seq2seq) |
| `src/models/vae.py` | `VAESynthesizer` |
| `src/models/trainer.py` | `ModelTrainer` training loops |
| `src/visualization/visualize.py` | Plots, Griffin-Lim, metrics displays |
| `tests/test_*.py` | Smoke tests |
| `notebooks/01-04*.ipynb` | Thin experiment runners |

---

## Phase 0: Project scaffold

### Task 0: `pyproject.toml` and package layout

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/visualization/__init__.py`
- Create: `data/processed/.gitkeep` (optional; dir created at runtime)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "nature-sound-generator"
version = "0.1.0"
description = "ESC-50 PyTorch lab: CNN, LSTM detection, VAE"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "torchaudio>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "pandas>=2.2.0",
    "jupyter>=1.0.0",
    "librosa>=0.10.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Lock deps and sync**

Run:
```bash
uv sync --all-extras
```
Expected: `.venv` created, `uv.lock` written.

- [ ] **Step 3: Create empty package inits**

```python
# src/__init__.py
# src/data/__init__.py
# src/models/__init__.py
# src/visualization/__init__.py
```
(each file empty or single comment)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/
git commit -m "chore: add uv project scaffold and src package layout"
```

---

## Phase 1: Data layer

### Task 1: `src/config.py`

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
from pathlib import Path
import src.config as cfg

def test_project_root_is_directory():
    assert cfg.PROJECT_ROOT.is_dir()

def test_raw_audio_dir_under_root():
    assert cfg.RAW_AUDIO_DIR == cfg.PROJECT_ROOT / "ESC-50-master" / "audio"

def test_num_classes():
    assert cfg.NUM_CLASSES == 50
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 3: Implement `src/config.py`**

```python
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
```

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add project config and dataset path helpers"
```

---

### Task 2: Mel transform helper (shared)

**Files:**
- Create: `src/data/audio_transforms.py`
- Test: `tests/test_audio_transforms.py`

- [ ] **Step 1: Write failing test** (uses synthetic waveform)

```python
# tests/test_audio_transforms.py
import torch
from src.data.audio_transforms import waveform_to_mel_128, WaveformParams

def test_waveform_to_mel_128_shape():
    wav = torch.randn(1, 44100 * 5)
    mel = waveform_to_mel_128(wav, WaveformParams())
    assert mel.shape == (1, 128, 128)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/test_audio_transforms.py -v
```

- [ ] **Step 3: Implement**

```python
# src/data/audio_transforms.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from src import config


@dataclass(frozen=True)
class WaveformParams:
    sample_rate: int = config.SAMPLE_RATE
    n_mels: int = config.MEL_N_MELS
    n_fft: int = config.N_FFT
    hop_length: int = config.HOP_LENGTH


def _mel_transform(params: WaveformParams) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=params.sample_rate,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        n_mels=params.n_mels,
    )


def _normalize_and_resize(mel: torch.Tensor, size: int = config.MEL_SIZE) -> torch.Tensor:
    mel = mel.unsqueeze(0) if mel.dim() == 2 else mel
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = F.interpolate(mel.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
    return mel.squeeze(0)


def waveform_to_mel_128(waveform: torch.Tensor, params: WaveformParams | None = None) -> torch.Tensor:
    params = params or WaveformParams()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = _mel_transform(params)(waveform)
    return _normalize_and_resize(mel)


def waveform_to_mel_sequence(waveform: torch.Tensor, params: WaveformParams | None = None) -> torch.Tensor:
    """Returns [T, n_mels] for LSTM (no resize on time axis)."""
    params = params or WaveformParams()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = _mel_transform(params)(waveform).squeeze(0)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.transpose(0, 1)
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/data/audio_transforms.py tests/test_audio_transforms.py
git commit -m "feat: add shared mel spectrogram transforms"
```

---

### Task 3: `make_dataset.py`

**Files:**
- Create: `src/data/make_dataset.py`
- Test: `tests/test_make_dataset.py`

- [ ] **Step 1: Write failing test** (skip if no audio: `pytest.importorskip` pattern — use tmp wav)

```python
# tests/test_make_dataset.py
import csv
from pathlib import Path
import torch
import torchaudio
import pytest

from src.data.make_dataset import process_one_file, build_manifest_row

@pytest.fixture
def tiny_wav(tmp_path):
    wav_path = tmp_path / "test.wav"
    torchaudio.save(str(wav_path), torch.randn(1, 44100), 44100)
    return wav_path

def test_process_one_file_shape(tiny_wav, tmp_path):
    out = tmp_path / "test.pt"
    process_one_file(tiny_wav, out)
    t = torch.load(out, weights_only=True)
    assert t.shape == (1, 128, 128)

def test_build_manifest_row():
    row = build_manifest_row("a.wav", 3, 1, "rain", tmp_path := Path("data/processed/a.pt"))
    assert row["target"] == 3
    assert row["fold"] == 1
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `src/data/make_dataset.py`**

```python
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch
import torchaudio

from src import config
from src.data.audio_transforms import waveform_to_mel_128, WaveformParams

logger = logging.getLogger(__name__)


def build_manifest_row(
    filename: str, target: int, fold: int, category: str, pt_path: Path
) -> dict:
    return {
        "filename": filename,
        "target": target,
        "fold": fold,
        "category": category,
        "path": str(pt_path.relative_to(config.PROJECT_ROOT)),
    }


def process_one_file(wav_path: Path, out_path: Path, params: WaveformParams | None = None) -> None:
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != config.SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)
    mel = waveform_to_mel_128(waveform, params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mel, out_path)


def should_skip(wav_path: Path, out_path: Path) -> bool:
    if not out_path.exists():
        return False
    return out_path.stat().st_mtime >= wav_path.stat().st_mtime


def run(min_success_ratio: float = 0.9) -> list[dict]:
    config.require_dataset()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    total = 0
    ok = 0

    with open(config.META_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        total += 1
        filename = row["filename"]
        wav_path = config.RAW_AUDIO_DIR / filename
        stem = Path(filename).stem
        out_path = config.PROCESSED_DIR / f"{stem}.pt"
        try:
            if not wav_path.exists():
                logger.warning("Missing %s", wav_path)
                continue
            if should_skip(wav_path, out_path):
                ok += 1
            else:
                process_one_file(wav_path, out_path)
                ok += 1
            manifest.append(
                build_manifest_row(
                    filename,
                    int(row["target"]),
                    int(row["fold"]),
                    row["category"],
                    out_path,
                )
            )
        except Exception as e:
            logger.warning("Failed %s: %s", filename, e)

    if ok / max(total, 1) < min_success_ratio:
        raise RuntimeError(f"Processed only {ok}/{total} files; need >={min_success_ratio:.0%}")

    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote %d entries to %s", len(manifest), config.MANIFEST_PATH)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/test_make_dataset.py -v
```

- [ ] **Step 5: Run full preprocess (manual, requires ESC-50)**

```bash
uv run python -m src.data.make_dataset
```
Expected: `data/processed/manifest.json` and ~2000 `.pt` files.

- [ ] **Step 6: Commit**

```bash
git add src/data/make_dataset.py tests/test_make_dataset.py
git commit -m "feat: add ESC-50 preprocessing to mel tensors"
```

**Phase 1 gate:** User confirms `make_dataset` ran successfully on real data.

---

### Task 4: `dataloaders.py`

**Files:**
- Create: `src/data/dataloaders.py`
- Test: `tests/test_dataloaders.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataloaders.py
import json
import torch
import pytest
from pathlib import Path
from unittest.mock import patch

from src.data.dataloaders import (
    ESC50Dataset,
    LSTMSyntheticDataset,
    fold_split,
    collate_lstm,
)

def test_fold_split():
    records = [{"fold": i} for i in [1, 2, 3, 4, 5]]
    train, val, test = fold_split(records, test_fold=5, val_fold=4)
    assert len(test) == 1
    assert all(r["fold"] == 5 for r in test)
    assert all(r["fold"] == 4 for r in val)

def test_collate_lstm():
    batch = [(torch.randn(10, 128), torch.randint(0, 2, (10,)).float())] * 2
    mels, labels = collate_lstm(batch)
    assert mels.shape[0] == 2
    assert labels.shape == mels.shape[:2]
```

Add integration test `test_lstm_labels_match_mel_time` after implementation (manifest + fake wav paths mocked).

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `src/data/dataloaders.py`**

Key exports:

```python
def fold_split(records, test_fold=5, val_fold=4) -> tuple[list, list, list]: ...

class ESC50Dataset(torch.utils.data.Dataset):
    # __getitem__ -> (tensor [1,128,128], label)

class LSTMSyntheticDataset(torch.utils.data.Dataset):
    # stitch 30s waveform from manifest-filtered clips
    # __getitem__ -> (mel [T,128], labels [T])

def collate_lstm(batch):  # pad sequences to max T in batch

def get_dataloaders(task: str, batch_size: int, test_fold: int = 5):
    # load manifest.json
    # task in {"cnn","vae","lstm"}
    # return train_loader, val_loader, test_loader
```

**LSTM synthesis algorithm (implement exactly):**

1. Load manifest; filter by split fold.
2. Partition records: `target_pool` (target==14), `other_pool` (target!=14).
3. Build `duration_samples = 30 * SAMPLE_RATE`.
4. Fill timeline left-to-right with segments until duration reached:
   - With prob 0.4 cumulative target time: pick segment from `target_pool` (loop/repeat 5s clips), label samples = 1.
   - Else 50/50 silence (zeros) or clip from `other_pool`, label = 0.
   - Min target segment 0.5 s.
5. `waveform_to_mel_sequence` → `[T, 128]`.
6. Map sample labels to frames: for each frame index `t`, `label[t] = 1` if midpoint sample in target region else `0`.

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/test_dataloaders.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/data/dataloaders.py tests/test_dataloaders.py
git commit -m "feat: add ESC50 and LSTM synthetic dataloaders with fold splits"
```

**End Phase 1** — ask user before Phase 2.

---

## Phase 2: Models

### Task 5: `CNNClassifier`

**Files:**
- Create: `src/models/cnn.py`
- Test: `tests/test_models.py` (CNN section)

- [ ] **Step 1: Failing test**

```python
def test_cnn_output_shape():
    from src.models.cnn import CNNClassifier
    model = CNNClassifier()
    x = torch.randn(4, 1, 128, 128)
    assert model(x).shape == (4, 50)
```

- [ ] **Step 2: Implement**

```python
# src/models/cnn.py
import torch.nn as nn
from src import config

class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

- [ ] **Step 3: pytest PASS, commit** `feat: add CNN classifier for 50 ESC-50 classes`

---

### Task 6: `AudioLSTMDetector`

- [ ] **Step 1: Test** `model(x).shape == (2, 17, 1)` for `x` shape `(2, 17, 128)`

- [ ] **Step 2: Implement**

```python
class AudioLSTMDetector(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)
```

- [ ] **Step 3: pytest PASS, commit** `feat: add seq2seq LSTM frame detector`

---

### Task 7: `VAESynthesizer`

- [ ] **Step 1: Test** forward returns `recon, mu, logvar` with `recon.shape == (2, 1, 128, 128)`

- [ ] **Step 2: Implement** Encoder (Conv2d stack) → `mu`, `logvar`; `reparameterize`; Decoder (ConvTranspose2d stack). Latent dim e.g. 128.

- [ ] **Step 3: pytest PASS, commit** `feat: add VAE synthesizer for mel spectrograms`

**End Phase 2** — user gate.

---

## Phase 3: Trainer

### Task 8: `ModelTrainer`

**Files:**
- Create: `src/models/trainer.py`
- Test: `tests/test_trainer.py` (one batch overfit smoke)

- [ ] **Step 1: Test** instantiate trainer, run `train_classifier` 1 epoch on tiny random tensor — no crash

- [ ] **Step 2: Implement**

```python
class ModelTrainer:
    def __init__(self, model, device=None, lr=config.LEARNING_RATE):
        self.model = model.to(device or config.get_device())
        self.device = device or config.get_device()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_classifier(self, train_loader, val_loader, epochs): ...
    def train_lstm_detector(self, train_loader, val_loader, epochs): ...
    def train_vae(self, train_loader, val_loader, epochs): ...
```

- `train_classifier`: `CrossEntropyLoss`, accuracy = `(pred==y).float().mean()`
- `train_lstm`: `BCEWithLogitsLoss`, squeeze logits/labels; frame accuracy
- `train_vae`: `recon_loss + kl_loss`, log both

- [ ] **Step 3: pytest PASS, commit** `feat: add ModelTrainer with CNN/LSTM/VAE loops`

**End Phase 3** — user gate.

---

## Phase 4: Visualization

### Task 9: `visualize.py`

**Files:**
- Create: `src/visualization/visualize.py`

- [ ] **Step 1: Implement functions (manual smoke in REPL acceptable)**

```python
def plot_training_history(history: dict, save_path: Path | None = None): ...
def plot_confusion_matrix(y_true, y_pred, class_names=None): ...
def plot_roc_curve(y_true, y_score): ...
def plot_spectrogram_pair(original: Tensor, reconstructed: Tensor, save_path=None): ...
def mel_to_audio(mel: Tensor, params: WaveformParams) -> Tensor:
    # Griffin-Lim via torchaudio.transforms.GriffinLim
```

- [ ] **Step 2: Commit** `feat: add training and evaluation visualization helpers`

**End Phase 4** — user gate.

---

## Phase 5: Notebooks

### Task 10: `01-data-exploration.ipynb`

- [ ] Import `config`, load `manifest.json`, show class distribution, plot 3 sample mels from `ESC50Dataset`
- [ ] Commit `docs: add data exploration notebook`

### Task 11: `02-cnn-classification.ipynb`

- [ ] `get_dataloaders("cnn", 32)`, `CNNClassifier`, `ModelTrainer.train_classifier`, confusion matrix via `visualize`

### Task 12: `03-lstm-detection.ipynb`

- [ ] `get_dataloaders("lstm", 8)`, train LSTM, plot ROC + one timeline of `labels` vs `sigmoid(logits)`

### Task 13: `04-vae-synthesis.ipynb`

- [ ] Train VAE, `plot_spectrogram_pair`, optional `mel_to_audio` playback

- [ ] Final commit `docs: add training notebooks for CNN LSTM VAE`

**End Phase 5** — project complete pending user review.

---

## Spec coverage checklist

| Spec requirement | Plan task |
|------------------|-----------|
| uv / pyproject | Task 0 |
| config paths, device | Task 1 |
| make_dataset → .pt + manifest | Task 3 |
| ESC50Dataset, fold split | Task 4 |
| LSTM 30s synthetic seq2seq labels | Task 4 |
| CNN 50-class | Task 5 |
| LSTM [B,T,1] | Task 6 |
| VAE mu/logvar/reparam | Task 7 |
| ModelTrainer 3 modes | Task 8 |
| Griffin-Lim, CM, ROC, spectrogram pair | Task 9 |
| Notebooks 01-04 | Tasks 10-13 |
| gitignore / README dataset | Done (pre-plan) |
| Phase gates | After Tasks 4, 7, 8, 9, 13 |

---

## Verification commands (end-to-end)

```bash
uv sync --all-extras
uv run python -m src.data.make_dataset
uv run pytest -v
uv run jupyter lab
```

Expected: all tests pass; notebooks run after preprocessing.
