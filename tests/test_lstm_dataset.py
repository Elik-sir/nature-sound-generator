import json

import torch


def _make_manifest():
    records = []
    for i, fold in enumerate([1, 2, 3, 4, 5]):
        records.append(
            {
                "filename": f"target_{i}.wav",
                "target": 14,
                "fold": fold,
                "category": "rain",
                "path": f"processed/target_{i}.pt",
            }
        )
        records.append(
            {
                "filename": f"other_{i}.wav",
                "target": i,
                "fold": fold,
                "category": "other",
                "path": f"processed/other_{i}.pt",
            }
        )
    return records


def test_collate_lstm_shapes():
    from src.data.dataloaders import collate_lstm

    batch = [
        (torch.randn(10, 128), torch.zeros(10)),
        (torch.randn(12, 128), torch.ones(12)),
    ]
    mels, labels = collate_lstm(batch)
    assert mels.shape == (2, 12, 128)
    assert labels.shape == (2, 12)


def test_lstm_dataset_labels_binary_and_aligned(tmp_path, monkeypatch):
    import src.config as cfg
    import src.data.dataloaders as dl
    from src.data.dataloaders import LSTMSyntheticDataset, fold_split

    manifest = _make_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(cfg, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "SAMPLE_RATE", 100)
    monkeypatch.setattr(cfg, "LSTM_DURATION_S", 2)
    monkeypatch.setattr(cfg, "HOP_LENGTH", 10)
    monkeypatch.setattr(cfg, "N_FFT", 20)

    def fake_load_waveform(_):
        return torch.randn(1, 400), 100

    def fake_resample_if_needed(waveform, _in_sr, _target_sr):
        return waveform

    def fake_waveform_to_mel_sequence(waveform):
        t = waveform.shape[-1] // 10
        return torch.randn(t, 128)

    monkeypatch.setattr(dl, "load_waveform", fake_load_waveform)
    monkeypatch.setattr(dl, "resample_if_needed", fake_resample_if_needed)
    monkeypatch.setattr(dl, "waveform_to_mel_sequence", fake_waveform_to_mel_sequence)

    train, _, _ = fold_split(manifest, test_fold=5, val_fold=4)
    ds = LSTMSyntheticDataset(train, root=tmp_path)
    mel, labels = ds[0]
    assert mel.shape[0] == labels.shape[0]
    assert mel.shape[1] == 128
    assert set(labels.unique().tolist()).issubset({0.0, 1.0})


def test_get_dataloaders_lstm(tmp_path, monkeypatch):
    import src.config as cfg
    import src.data.dataloaders as dl
    from src.data.dataloaders import get_dataloaders

    manifest = _make_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(cfg, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "SAMPLE_RATE", 100)
    monkeypatch.setattr(cfg, "LSTM_DURATION_S", 2)
    monkeypatch.setattr(cfg, "HOP_LENGTH", 10)
    monkeypatch.setattr(cfg, "N_FFT", 20)
    monkeypatch.setattr(cfg, "BATCH_SIZE_LSTM", 2)

    def fake_load_waveform(_):
        return torch.randn(1, 400), 100

    def fake_resample_if_needed(waveform, _in_sr, _target_sr):
        return waveform

    def fake_waveform_to_mel_sequence(waveform):
        t = waveform.shape[-1] // 10
        return torch.randn(t, 128)

    monkeypatch.setattr(dl, "load_waveform", fake_load_waveform)
    monkeypatch.setattr(dl, "resample_if_needed", fake_resample_if_needed)
    monkeypatch.setattr(dl, "waveform_to_mel_sequence", fake_waveform_to_mel_sequence)

    train_loader, val_loader, test_loader = get_dataloaders("lstm", batch_size=2)
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    mels, labels = next(iter(train_loader))
    assert mels.dim() == 3
    assert labels.dim() == 2
    assert mels.shape[0] == labels.shape[0]
    assert mels.shape[1] == labels.shape[1]
    assert mels.shape[2] == 128


def test_export_lstm_debug_samples_writes_audio_and_labels(tmp_path, monkeypatch):
    import src.config as cfg
    import src.data.dataloaders as dl
    from src.data.dataloaders import export_lstm_debug_samples, fold_split

    manifest = _make_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(cfg, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "SAMPLE_RATE", 100)
    monkeypatch.setattr(cfg, "LSTM_DURATION_S", 2)
    monkeypatch.setattr(cfg, "HOP_LENGTH", 10)
    monkeypatch.setattr(cfg, "N_FFT", 20)

    def fake_load_waveform(_):
        return torch.randn(1, 400), 100

    def fake_resample_if_needed(waveform, _in_sr, _target_sr):
        return waveform

    def fake_waveform_to_mel_sequence(waveform):
        t = waveform.shape[-1] // 10
        return torch.randn(t, 128)

    monkeypatch.setattr(dl, "load_waveform", fake_load_waveform)
    monkeypatch.setattr(dl, "resample_if_needed", fake_resample_if_needed)
    monkeypatch.setattr(dl, "waveform_to_mel_sequence", fake_waveform_to_mel_sequence)

    train, _, _ = fold_split(manifest, test_fold=5, val_fold=4)
    out_dir = tmp_path / "lstm_debug"
    paths = export_lstm_debug_samples(train, out_dir=out_dir, n_samples=2, root=tmp_path)

    assert len(paths) == 2
    for wav_path in paths:
        assert wav_path.exists()
        labels_path = wav_path.with_suffix(".labels.pt")
        meta_path = wav_path.with_suffix(".json")
        assert labels_path.exists()
        assert meta_path.exists()
