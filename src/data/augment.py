from __future__ import annotations

import random

import torch

from src import config


def _freq_mask(spec: torch.Tensor, max_width: int) -> torch.Tensor:
    _, freq, _time = spec.shape
    width = random.randint(1, min(max_width, freq))
    start = random.randint(0, freq - width)
    spec = spec.clone()
    spec[:, start : start + width, :] = 0
    return spec


def _time_mask(spec: torch.Tensor, max_width: int) -> torch.Tensor:
    _c, _freq, time = spec.shape
    width = random.randint(1, min(max_width, time))
    start = random.randint(0, time - width)
    spec = spec.clone()
    spec[:, :, start : start + width] = 0
    return spec


def augment_mel(spec: torch.Tensor) -> torch.Tensor:
    """SpecAugment-style augmentation for training. Input shape [1, H, W]."""
    if spec.dim() == 2:
        spec = spec.unsqueeze(0)

    if random.random() < config.AUG_FREQ_MASK_PROB:
        spec = _freq_mask(spec, config.AUG_FREQ_MASK_MAX)
    if random.random() < config.AUG_TIME_MASK_PROB:
        spec = _time_mask(spec, config.AUG_TIME_MASK_MAX)
    if random.random() < config.AUG_NOISE_PROB:
        spec = spec + config.AUG_NOISE_STD * torch.randn_like(spec)
    gain = random.uniform(config.AUG_GAIN_MIN, config.AUG_GAIN_MAX)
    spec = spec * gain
    return spec


def augment_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Lightweight waveform augmentation for ESC-50 training."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    augmented = waveform.clone()

    # Random circular time shift up to configured max seconds.
    max_shift = int(config.AUDIO_SHIFT_MAX_SECONDS * sample_rate)
    if max_shift > 0:
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            augmented = torch.roll(augmented, shifts=shift, dims=-1)

    # Random gain.
    gain = random.uniform(config.AUDIO_GAIN_MIN, config.AUDIO_GAIN_MAX)
    augmented = augmented * gain

    # Additive gaussian noise.
    if config.AUDIO_NOISE_STD > 0:
        augmented = augmented + config.AUDIO_NOISE_STD * torch.randn_like(augmented)

    # Keep waveform in valid audio range.
    return augmented.clamp(-1.0, 1.0)
