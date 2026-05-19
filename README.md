# Nature Sound Generator

PyTorch lab for ESC-50: CNN classification, LSTM frame-level detection, and VAE spectrogram synthesis.

## Dataset setup

This repository does **not** include audio files. Download the dataset and place it locally:

1. Download **ESC-50** from the official repository:  
   https://github.com/karolpiczak/ESC-50  
   (see releases / README for the archive with `audio/` and `meta/`.)

2. Extract the archive so the project root contains:

   ```text
   nature-sound-generator/
   └── ESC-50-master/
       ├── audio/          # ~2000 .wav files
       └── meta/
           └── esc50.csv
   ```

3. Run preprocessing (after `uv sync`):

   ```bash
   uv run python -m src.data.make_dataset
   ```

Processed tensors are written to `data/processed/` (also gitignored).

## Environment

```bash
uv sync --all-extras
```

### GPU (CUDA)

This project pins **PyTorch with CUDA 12.6** wheels (`cu126`) for NVIDIA GPUs. After `uv sync`, verify:

```bash
uv run python -c "from src.config import get_device, describe_device; print(describe_device(get_device()))"
```

Expected output includes your GPU name, e.g. `cuda:0 (NVIDIA GeForce RTX 4060 Ti, torch 2.x+cu126)`.

If you see `cpu` or `cuda_available False`, reinstall deps:

```bash
uv sync --reinstall-package torch --reinstall-package torchaudio
```

Training uses GPU automatically. To force CPU (debug only): `set FORCE_CPU=1` (Windows) or `export FORCE_CPU=1` (Linux/macOS).

Optional: `CUDA_DEVICE=0` to pick a GPU index.

### Model checkpoints (versioned)

Weights are saved under `models/checkpoints/{cnn|lstm|vae}/` (gitignored):

- `v001_YYYYMMDD_HHMMSS/checkpoint.pt` — each save (per epoch by default)
- `best.pt` / `latest.pt` — quick load symlinks (copies)
- `registry.json` — all versions with `val_acc`, `val_loss`, epoch

```bash
uv run python -c "from src.models import checkpoints; print(checkpoints.list_versions('cnn'))"
```

In code: `checkpoints.load_best("cnn", model)` before evaluation to skip retraining.

See `docs/superpowers/specs/2026-05-19-nature-sound-generator-design.md` for full architecture and implementation phases.
