import torch

from src.data.augment import augment_mel


def test_augment_preserves_shape():
    spec = torch.randn(1, 128, 128)
    out = augment_mel(spec)
    assert out.shape == (1, 128, 128)
