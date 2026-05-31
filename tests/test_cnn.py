import torch

from src.models.cnn import CNNClassifier


def test_cnn_output_shape():
    model = CNNClassifier()
    x = torch.randn(4, 1, 128, 128)
    assert model(x).shape == (4, 50)


def test_cnn_uses_required_blocks_and_no_softmax():
    model = CNNClassifier()
    module_types = {type(m).__name__ for m in model.modules()}
    assert "Conv2d" in module_types
    assert "BatchNorm2d" in module_types
    assert "ReLU" in module_types
    assert "MaxPool2d" in module_types
    assert "Softmax" not in module_types
