import torch

from src.data.audio_transforms import (
    WaveformParams,
    waveform_to_log_mel_vae,
    waveform_to_mel_128,
)


def test_waveform_to_mel_128_shape():
    wav = torch.randn(1, 44100 * 5)
    mel = waveform_to_mel_128(wav, WaveformParams())
    assert mel.shape == (1, 128, 128)


def test_waveform_to_log_mel_vae_shape_and_finite():
    wav = torch.randn(1, 44100 * 5)
    log_mel = waveform_to_log_mel_vae(wav, WaveformParams())
    assert log_mel.shape == (1, 128, 128)
    assert torch.isfinite(log_mel).all()
    assert float(log_mel.min()) >= -12.0 - 1e-5
    assert float(log_mel.max()) <= 2.0 + 1e-5
