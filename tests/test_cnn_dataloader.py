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


def test_get_dataloaders_vae_filters_bird_class(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    mel = torch.randn(1, 128, 128)
    for i in range(10):
        torch.save(mel, processed / f"clip_{i}.pt")

    manifest = []
    folds = [1, 2, 3, 4, 5] * 2
    targets = [14, 1, 14, 2, 14, 3, 14, 4, 14, 5]
    for i, (fold, target) in enumerate(zip(folds, targets)):
        manifest.append(
            {
                "filename": f"{i}.wav",
                "target": target,
                "fold": fold,
                "category": "test",
                "path": f"processed/clip_{i}.pt",
            }
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    import src.config as cfg
    from src.data.dataloaders import get_dataloaders

    monkeypatch.setattr(cfg, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "VAE_TARGET_CLASS", 14, raising=False)

    train_loader, val_loader, test_loader = get_dataloaders("vae", batch_size=4)

    train_labels = next(iter(train_loader))[1]
    val_labels = next(iter(val_loader))[1]
    test_labels = next(iter(test_loader))[1]

    assert torch.all(train_labels == 14)
    assert torch.all(val_labels == 14)
    assert torch.all(test_labels == 14)


def test_vae_train_loader_applies_random_crop(tmp_path, monkeypatch):
    import src.config as cfg
    import src.data.dataloaders as dl
    from src.data.dataloaders import get_dataloaders

    processed = tmp_path / "processed"
    processed.mkdir()
    mel = torch.randn(1, 128, 128)
    for i in range(5):
        torch.save(mel, processed / f"clip_{i}.pt")
        wav_path = tmp_path / "ESC-50-master" / "audio"
        wav_path.mkdir(parents=True, exist_ok=True)
        (wav_path / f"{i}.wav").write_bytes(b"fake")

    manifest = [
        {
            "filename": f"{i}.wav",
            "target": 14,
            "fold": fold,
            "category": "bird",
            "path": f"processed/clip_{i}.pt",
        }
        for i, fold in enumerate([1, 2, 3, 4, 5])
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(cfg, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "SAMPLE_RATE", 10)
    monkeypatch.setattr(cfg, "VAE_TARGET_CLASS", 14, raising=False)
    monkeypatch.setattr(cfg, "VAE_CROP_SECONDS", 2.0, raising=False)

    def fake_load_waveform(_):
        return torch.randn(1, 50), 10

    def fake_resample_if_needed(w, _sr_in, _sr_out):
        return w

    crop_lengths: list[int] = []

    def fake_waveform_to_log_mel_vae(w):
        crop_lengths.append(int(w.shape[-1]))
        return torch.randn(1, 128, 128)

    monkeypatch.setattr(dl, "load_waveform", fake_load_waveform)
    monkeypatch.setattr(dl, "resample_if_needed", fake_resample_if_needed)
    monkeypatch.setattr(dl, "waveform_to_log_mel_vae", fake_waveform_to_log_mel_vae)

    train_loader, val_loader, _ = get_dataloaders("vae", batch_size=2)
    _ = next(iter(train_loader))
    _ = next(iter(val_loader))

    # Train sample should be cropped to 2s * 10Hz = 20 samples.
    assert 20 in crop_lengths
    # Validation keeps full length (50 samples), no train-time crop.
    assert 50 in crop_lengths
