from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src import config
from src.data.make_dataset import build_manifest_row, process_one_file


def test_process_one_file_shape(tmp_path):
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), np.random.randn(44100).astype(np.float32), 44100)
    out = tmp_path / "test.pt"
    process_one_file(wav_path, out)
    tensor = torch.load(out, weights_only=True)
    assert tensor.shape == (1, 128, 128)


def test_build_manifest_row():
    pt_path = config.PROJECT_ROOT / "data" / "processed" / "a.pt"
    row = build_manifest_row("a.wav", 3, 1, "rain", pt_path)
    assert row["target"] == 3
    assert row["fold"] == 1
