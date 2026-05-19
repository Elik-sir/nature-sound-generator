from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def load_waveform(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio as float tensor [channels, time]. Uses soundfile (no torchcodec)."""
    data, sample_rate = sf.read(str(path), always_2d=True)
    waveform = torch.from_numpy(data.T).float()
    return waveform, int(sample_rate)


def resample_if_needed(
    waveform: torch.Tensor, sample_rate: int, target_rate: int
) -> torch.Tensor:
    if sample_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, target_rate)
