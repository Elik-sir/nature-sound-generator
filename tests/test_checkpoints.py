import json

import torch

from src.models import checkpoints
from src.models.cnn import CNNClassifier


def test_save_best_overwrites_single_file(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(checkpoints.config, "PROJECT_ROOT", root)

    model = CNNClassifier()
    checkpoints.save_best(
        "cnn",
        model,
        epoch=1,
        metrics={"val_acc": 0.4, "val_loss": 1.2},
    )
    checkpoints.save_best(
        "cnn",
        model,
        epoch=5,
        metrics={"val_acc": 0.55, "val_loss": 0.9},
    )

    best_path = root / "models" / "checkpoints" / "cnn" / "best.pt"
    assert best_path.exists()
    assert len(list((root / "models" / "checkpoints" / "cnn").glob("v*"))) == 0

    registry = json.loads(
        (root / "models" / "checkpoints" / "cnn" / "registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["epoch"] == 5
    assert registry["metrics"]["val_loss"] == 0.9

    model2 = CNNClassifier()
    payload = checkpoints.load_best("cnn", model2)
    assert payload is not None
    assert payload["epoch"] == 5
    assert len(checkpoints.list_versions("cnn")) == 1
