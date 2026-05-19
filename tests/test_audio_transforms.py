import torch

from src.data.audio_transforms import WaveformParams, waveform_to_mel_128


def test_waveform_to_mel_128_shape():
    wav = torch.randn(1, 44100 * 5)
    mel = waveform_to_mel_128(wav, WaveformParams())
    assert mel.shape == (1, 128, 128)
