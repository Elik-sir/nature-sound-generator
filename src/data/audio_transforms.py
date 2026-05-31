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
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = F.interpolate(
        mel.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return mel.squeeze(0)


def waveform_to_mel_128(
    waveform: torch.Tensor, params: WaveformParams | None = None
) -> torch.Tensor:
    params = params or WaveformParams()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = _mel_transform(params)(waveform)
    return _normalize_and_resize(mel)


def waveform_to_mel_sequence(
    waveform: torch.Tensor, params: WaveformParams | None = None
) -> torch.Tensor:
    """Returns [T, n_mels] for LSTM (no resize on time axis)."""
    params = params or WaveformParams()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = _mel_transform(params)(waveform).squeeze(0)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.transpose(0, 1)


def waveform_to_log_mel_vae(
    waveform: torch.Tensor, params: WaveformParams | None = None
) -> torch.Tensor:
    """
    VAE preprocessing: log-mel resized to 128x128 without z-score normalization.
    Keeps amplitude structure more faithful for inversion during listening.
    """
    params = params or WaveformParams()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = _mel_transform(params)(waveform)
    log_mel = torch.log(mel.clamp_min(1e-6))
    log_mel = log_mel.clamp(min=config.VAE_LOG_MEL_MIN, max=config.VAE_LOG_MEL_MAX)
    log_mel = F.interpolate(
        log_mel.unsqueeze(0),
        size=(config.MEL_SIZE, config.MEL_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return log_mel.squeeze(0)
