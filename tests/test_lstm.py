import torch

from src.models.lstm import AudioLSTMDetector


def test_lstm_output_shape():
    model = AudioLSTMDetector()
    x = torch.randn(2, 17, 128)
    y = model(x)
    assert y.shape == (2, 17, 1)
