import torch

from src.data.augment import augment_waveform


def test_augment_waveform_preserves_shape_and_range():
    waveform = torch.randn(1, 44100)
    out = augment_waveform(waveform, sample_rate=44100)
    assert out.shape == waveform.shape
    assert torch.all(out <= 1.0)
    assert torch.all(out >= -1.0)
