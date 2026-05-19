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
uv sync
```

See `docs/superpowers/specs/2026-05-19-nature-sound-generator-design.md` for full architecture and implementation phases.
