import torch

from src.models.cnn import CNNClassifier


def test_cnn_output_shape():
    model = CNNClassifier()
    x = torch.randn(4, 1, 128, 128)
    assert model(x).shape == (4, 50)
