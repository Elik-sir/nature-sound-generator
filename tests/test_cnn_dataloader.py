import json
from pathlib import Path

import torch

from src.data.dataloaders import ESC50Dataset, fold_split


def test_fold_split():
    records = [{"fold": i} for i in [1, 2, 3, 4, 5]]
    train, val, test = fold_split(records, test_fold=5, val_fold=4)
    assert len(test) == 1
    assert all(r["fold"] == 5 for r in test)
    assert all(r["fold"] == 4 for r in val)
    assert len(train) == 3


def test_esc50_dataset_loads_pt(tmp_path):
    mel = torch.randn(1, 128, 128)
    pt_path = tmp_path / "sample.pt"
    torch.save(mel, pt_path)

    record = {
        "filename": "x.wav",
        "target": 7,
        "fold": 1,
        "category": "rain",
        "path": "sample.pt",
    }
    dataset = ESC50Dataset([record], root=tmp_path)
    spec, label = dataset[0]
    assert spec.shape == (1, 128, 128)
    assert label == 7


def test_get_dataloaders_cnn(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    mel = torch.randn(1, 128, 128)
    for i, fold in enumerate([1, 2, 3, 4, 5]):
        torch.save(mel, processed / f"clip_{i}.pt")

    manifest = [
        {
            "filename": f"{i}.wav",
            "target": i,
            "fold": fold,
            "category": "test",
            "path": f"processed/clip_{i}.pt",
        }
        for i, fold in enumerate([1, 2, 3, 4, 5])
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    import src.config as cfg

    monkeypatch.setattr(cfg, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)

    from src.data.dataloaders import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        "cnn", batch_size=2
    )
    assert len(train_loader) == 2
    assert len(val_loader) == 1
    assert len(test_loader) == 1

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (2, 1, 128, 128)
    assert batch_y.shape == (2,)
