# Nature Sound Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CCDS Lite PyTorch lab on ESC-50 with CNN classification, LSTM frame-level detection on synthetic 30 s tracks, and VAE mel synthesis.

**Architecture:** Hybrid data pipeline — preprocess 5 s clips to `data/processed/*.pt`; CNN/VAE read cached mels; LSTM stitches waveforms at runtime with per-frame labels. Fold-based splits, `uv` for deps, no audio in git.

**Tech Stack:** Python 3.11+, PyTorch, torchaudio, uv, scikit-learn, matplotlib, seaborn, pytest

**Design spec:** `docs/superpowers/specs/2026-05-19-nature-sound-generator-design.md`

**Gate:** Stop after each **model phase** (CNN → LSTM → VAE) and wait for user approval. Debug and validate one model end-to-end before starting the next.

---

## Execution roadmap (model-centric)

| Phase | Focus | Deliverables | Tests | Notebook |
|-------|--------|--------------|-------|----------|
| **0** | Scaffold | `pyproject.toml`, `uv.lock`, package inits | — | — |
| **1** | **CNN** | Data pipeline for clips, `CNNClassifier`, `train_classifier`, CNN plots | `test_config`, `test_make_dataset`, `test_cnn_dataloader`, `test_cnn`, `test_cnn_trainer` | `02-cnn-classification.ipynb` |
| **2** | **LSTM** | `LSTMSyntheticDataset`, `AudioLSTMDetector`, `train_lstm_detector`, ROC plots | `test_lstm_dataset`, `test_lstm`, `test_lstm_trainer` | `03-lstm-detection.ipynb` |
| **3** | **VAE** | `VAESynthesizer`, `train_vae`, spectrogram comparison | `test_vae`, `test_vae_trainer` | `04-vae-synthesis.ipynb` |
| **4** | Integration | `01-data-exploration.ipynb`, full `pytest`, README polish | all tests green | `01-data-exploration.ipynb` |

**Incremental `ModelTrainer`:** add only the method needed per phase (`train_classifier` → `train_lstm_detector` → `train_vae`). Do not implement VAE/LSTM training loops while still debugging CNN.

**Incremental `dataloaders.py`:** Phase 1 adds `ESC50Dataset` + `get_dataloaders(task="cnn"|"vae")`. Phase 2 adds `LSTMSyntheticDataset`, `collate_lstm`, and `task="lstm"`.

**Incremental `visualize.py`:** Phase 1 — training curves + confusion matrix. Phase 2 — ROC. Phase 3 — spectrogram pair + Griffin-Lim.

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
| `tests/test_cnn*.py` | CNN phase tests |
| `tests/test_lstm*.py` | LSTM phase tests |
| `tests/test_vae*.py` | VAE phase tests |
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

## Phase 1: CNN track (data + model + train + debug)

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

**CNN data gate:** User confirms `make_dataset` ran successfully on real ESC-50.

---

### Task 4: `dataloaders.py` (CNN / VAE only — no LSTM yet)

**Files:**
- Create: `src/data/dataloaders.py`
- Test: `tests/test_cnn_dataloader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cnn_dataloader.py
import torch
from src.data.dataloaders import ESC50Dataset, fold_split

def test_fold_split():
    records = [{"fold": i} for i in [1, 2, 3, 4, 5]]
    train, val, test = fold_split(records, test_fold=5, val_fold=4)
    assert len(test) == 1
    assert all(r["fold"] == 5 for r in test)
    assert all(r["fold"] == 4 for r in val)
```

Add `test_esc50_dataset_loads_pt` using a fixture manifest + tiny `.pt` file.

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement CNN slice of `dataloaders.py`**

```python
def fold_split(records, test_fold=5, val_fold=4) -> tuple[list, list, list]: ...

class ESC50Dataset(torch.utils.data.Dataset):
    # __getitem__ -> (tensor [1,128,128], label int)

def get_dataloaders(task: str, batch_size: int, test_fold: int = 5):
    # Phase 1: assert task in ("cnn", "vae")
    # load manifest.json, split by fold, return train/val/test DataLoaders
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/test_cnn_dataloader.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/data/dataloaders.py tests/test_cnn_dataloader.py
git commit -m "feat: add ESC50 dataset and fold splits for CNN"
```

---

### Task 5: `CNNClassifier`

**Files:**
- Create: `src/models/cnn.py`
- Test: `tests/test_cnn.py`

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

```bash
uv run pytest tests/test_cnn.py -v
```

---

### Task 6: `ModelTrainer.train_classifier` (CNN only)

**Files:**
- Create: `src/models/trainer.py` (skeleton + `train_classifier` only)
- Test: `tests/test_cnn_trainer.py`

- [ ] **Step 1: Test** — 1-epoch smoke on `DataLoader` with 2 random batches

- [ ] **Step 2: Implement** `ModelTrainer` with `train_classifier` only (`CrossEntropyLoss`, epoch loss/accuracy logs)

- [ ] **Step 3: pytest PASS, commit** `feat: add CNN training loop`

---

### Task 7: CNN visualization

**Files:**
- Create: `src/visualization/visualize.py` (partial)
- Functions: `plot_training_history`, `plot_confusion_matrix`

- [ ] **Step 1: Implement CNN plot helpers**

- [ ] **Step 2: Commit** `feat: add CNN training and confusion matrix plots`

---

### Task 8: CNN notebook + manual debug

**Files:**
- Create: `notebooks/02-cnn-classification.ipynb`

- [ ] Run `get_dataloaders("cnn", 32)`, train 20 epochs, plot curves + confusion matrix on test fold
- [ ] Commit `docs: add CNN classification notebook`

**End Phase 1 (CNN)** — user reviews accuracy/loss, approves before LSTM.

```bash
uv run pytest tests/test_config.py tests/test_make_dataset.py tests/test_cnn_dataloader.py tests/test_cnn.py tests/test_cnn_trainer.py -v
```

---

## Phase 2: LSTM track (seq2seq detection)

### Task 9: Extend `dataloaders.py` for LSTM

**Files:**
- Modify: `src/data/dataloaders.py`
- Test: `tests/test_lstm_dataset.py`

- [ ] **Step 1: Write failing tests** for `LSTMSyntheticDataset`, `collate_lstm`, `get_dataloaders("lstm", ...)`

```python
def test_collate_lstm():
    batch = [(torch.randn(10, 128), torch.zeros(10))] * 2
    mels, labels = collate_lstm(batch)
    assert mels.shape[0] == 2

def test_lstm_labels_binary_and_align():
    # mel.shape[0] == labels.shape[0], labels in {0.0, 1.0}
```

- [ ] **Step 2: Implement `LSTMSyntheticDataset`** (30 s waveform stitch, class 14, per-frame labels)

**LSTM synthesis algorithm (implement exactly):**

1. Load manifest; filter by split fold.
2. Partition: `target_pool` (target==14), `other_pool` (target!=14).
3. `duration_samples = 30 * SAMPLE_RATE`.
4. Stitch segments: ~40% timeline target (label 1), else silence/other (label 0); min target segment 0.5 s.
5. `waveform_to_mel_sequence` → `[T, 128]`.
6. Map sample-level regions to frame labels via hop midpoint.

- [ ] **Step 3: Extend `get_dataloaders` to support `task="lstm"`**

- [ ] **Step 4: pytest PASS, commit** `feat: add LSTM synthetic dataset and collate`

---

### Task 10: `AudioLSTMDetector`

**Files:**
- Create: `src/models/lstm.py`
- Test: `tests/test_lstm.py`

- [ ] **Step 1: Test** `model(x).shape == (2, 17, 1)` for `x` shape `(2, 17, 128)`

- [ ] **Step 2: Implement seq2seq LSTM** (`nn.LSTM` + per-step `Linear` → `[B, T, 1]`)

- [ ] **Step 3: pytest PASS, commit** `feat: add seq2seq LSTM frame detector`

---

### Task 11: `ModelTrainer.train_lstm_detector`

**Files:**
- Modify: `src/models/trainer.py`
- Test: `tests/test_lstm_trainer.py`

- [ ] **Step 1: Add `train_lstm_detector`** — `BCEWithLogitsLoss`, frame accuracy

- [ ] **Step 2: pytest PASS, commit** `feat: add LSTM training loop`

---

### Task 12: LSTM visualization + notebook

**Files:**
- Modify: `src/visualization/visualize.py` — add `plot_roc_curve`
- Create: `notebooks/03-lstm-detection.ipynb`

- [ ] Train LSTM 30 epochs, ROC on test frames, timeline plot (labels vs sigmoid(logits))
- [ ] Commit `docs: add LSTM detection notebook`

**End Phase 2 (LSTM)** — user gate.

```bash
uv run pytest tests/test_lstm_dataset.py tests/test_lstm.py tests/test_lstm_trainer.py -v
```

---

## Phase 3: VAE track (synthesis)

### Task 13: `VAESynthesizer`

**Files:**
- Create: `src/models/vae.py`
- Test: `tests/test_vae.py`

- [ ] **Step 1: Test** forward returns `recon, mu, logvar` with `recon.shape == (2, 1, 128, 128)`

- [ ] **Step 2: Implement** Encoder → `mu`, `logvar`; `reparameterize`; Decoder (`ConvTranspose2d`); latent dim 128

- [ ] **Step 3: pytest PASS, commit** `feat: add VAE synthesizer for mel spectrograms`

---

### Task 14: `ModelTrainer.train_vae`

**Files:**
- Modify: `src/models/trainer.py`
- Test: `tests/test_vae_trainer.py`

- [ ] **Step 1: Add `train_vae`** — MSE recon + KL, log both per epoch

- [ ] **Step 2: pytest PASS, commit** `feat: add VAE training loop`

---

### Task 15: VAE visualization + notebook

**Files:**
- Modify: `src/visualization/visualize.py` — `plot_spectrogram_pair`, `mel_to_audio` (Griffin-Lim)
- Create: `notebooks/04-vae-synthesis.ipynb`

- [ ] Train VAE 50 epochs on `get_dataloaders("vae", 32)`, compare input/recon spectrograms
- [ ] Commit `docs: add VAE synthesis notebook`

**End Phase 3 (VAE)** — user gate.

```bash
uv run pytest tests/test_vae.py tests/test_vae_trainer.py -v
```

---

## Phase 4: Integration

### Task 16: EDA notebook

- [ ] Create `notebooks/01-data-exploration.ipynb` — manifest stats, class distribution, sample spectrograms
- [ ] Commit `docs: add data exploration notebook`

### Task 17: Full verification

```bash
uv sync --all-extras
uv run python -m src.data.make_dataset
uv run pytest -v
```

**End Phase 4** — project complete pending user review.

---

## Spec coverage checklist

| Spec requirement | Plan phase |
|------------------|------------|
| uv / pyproject | Phase 0, Task 0 |
| config paths, device | Phase 1, Task 1 |
| make_dataset → .pt + manifest | Phase 1, Task 3 |
| ESC50Dataset, fold split | Phase 1, Task 4 |
| CNN 50-class + train + debug | Phase 1, Tasks 5–8 |
| LSTM 30s synthetic seq2seq labels | Phase 2, Task 9 |
| LSTM [B,T,1] + train + ROC | Phase 2, Tasks 10–12 |
| VAE mu/logvar/reparam + train | Phase 3, Tasks 13–15 |
| Griffin-Lim, spectrogram pair | Phase 3, Task 15 |
| Confusion matrix, training curves | Phase 1, Task 7 |
| Notebook 01 EDA | Phase 4, Task 16 |
| Notebooks 02–04 | Phases 1–3 |
| gitignore / README dataset | Done (pre-plan) |
| Phase gates | After Phases 1, 2, 3 |

---

## Verification commands (end-to-end)

```bash
uv sync --all-extras
uv run python -m src.data.make_dataset
uv run pytest -v
uv run jupyter lab
```

Expected: all tests pass; notebooks run after preprocessing.
