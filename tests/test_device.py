import os

import torch

from src import config


def test_describe_device_cpu_when_forced(monkeypatch):
    monkeypatch.setenv("FORCE_CPU", "1")
    assert "cpu" in config.describe_device(config.get_device())


def test_get_device_cuda_when_available():
    if not torch.cuda.is_available():
        return
    device = config.get_device()
    assert device.type == "cuda"
    assert "cuda" in config.describe_device(device)
