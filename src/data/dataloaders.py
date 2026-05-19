from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src import config


def load_manifest(path: Path | None = None) -> list[dict]:
    manifest_path = path or config.MANIFEST_PATH
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def fold_split(
    records: list[dict],
    test_fold: int = config.TEST_FOLD,
    val_fold: int = config.VAL_FOLD,
) -> tuple[list[dict], list[dict], list[dict]]:
    test = [r for r in records if r["fold"] == test_fold]
    val = [r for r in records if r["fold"] == val_fold]
    train = [
        r for r in records if r["fold"] not in (test_fold, val_fold)
    ]
    return train, val, test


class ESC50Dataset(Dataset):
    def __init__(self, records: list[dict], root: Path | None = None):
        self.records = records
        self.root = root or config.PROJECT_ROOT

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        path = self.root / record["path"]
        spectrogram = torch.load(path, weights_only=True)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
        return spectrogram, int(record["target"])


def _make_loader(
    records: list[dict], batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        ESC50Dataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.get_num_workers(),
    )


def get_dataloaders(
    task: str,
    batch_size: int,
    test_fold: int = config.TEST_FOLD,
    val_fold: int = config.VAL_FOLD,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if task not in ("cnn", "vae"):
        raise ValueError(
            f"Phase 1 supports task 'cnn' or 'vae' only, got {task!r}. "
            "Use task='lstm' after LSTM phase is implemented."
        )

    records = load_manifest()
    train_records, val_records, test_records = fold_split(
        records, test_fold=test_fold, val_fold=val_fold
    )

    train_loader = _make_loader(train_records, batch_size, shuffle=True)
    val_loader = _make_loader(val_records, batch_size, shuffle=False)
    test_loader = _make_loader(test_records, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
