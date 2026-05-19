import json

import torch
from torch import nn

from src.models import checkpoints
from src.models.cnn import CNNClassifier


def test_save_and_load_best(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    ckpt_root = root / "models" / "checkpoints"
    monkeypatch.setattr(checkpoints.config, "PROJECT_ROOT", root)
    monkeypatch.setattr(checkpoints, "checkpoints_root", lambda: ckpt_root)

    model = CNNClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoints.save_checkpoint(
        "cnn",
        model,
        optimizer=opt,
        epoch=1,
        metrics={"val_acc": 0.5, "val_loss": 1.0},
        is_best=True,
    )

    model2 = CNNClassifier()
    payload = checkpoints.load_best("cnn", model2)
    assert payload is not None
    assert payload["version"] == 1

    registry_path = ckpt_root / "cnn" / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["best_version"] == 1
    assert len(registry["versions"]) == 1


def test_version_increments(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    ckpt_root = root / "models" / "checkpoints"
    monkeypatch.setattr(checkpoints.config, "PROJECT_ROOT", root)

    model = CNNClassifier()
    v1 = checkpoints.save_checkpoint(
        "cnn", model, epoch=1, metrics={"val_acc": 0.4}, is_best=True
    )
    v2 = checkpoints.save_checkpoint(
        "cnn", model, epoch=2, metrics={"val_acc": 0.6}, is_best=True
    )
    assert v2.version == v1.version + 1

    versions = checkpoints.list_versions("cnn")
    assert len(versions) == 2
    best = checkpoints.get_best_checkpoint("cnn")
    assert best is not None
    assert best.version == 2
    assert best.val_acc == 0.6
