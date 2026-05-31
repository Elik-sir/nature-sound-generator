import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import checkpoints
from src.models.lstm import AudioLSTMDetector
from src.models.trainer import ModelTrainer


def _make_lstm_loader(n_samples: int = 8, t_steps: int = 16) -> DataLoader:
    x = torch.randn(n_samples, t_steps, 128)
    y = torch.randint(0, 2, (n_samples, t_steps)).float()
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_train_lstm_detector_one_epoch():
    train_loader = _make_lstm_loader()
    val_loader = _make_lstm_loader()

    trainer = ModelTrainer(AudioLSTMDetector(), device=torch.device("cpu"), model_name="lstm")
    history = trainer.train_lstm_detector(train_loader, val_loader, epochs=1)

    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert len(history["train_acc"]) == 1
    assert len(history["val_acc"]) == 1


def test_predict_lstm_scores_shapes():
    loader = _make_lstm_loader(n_samples=4, t_steps=10)
    trainer = ModelTrainer(AudioLSTMDetector(), device=torch.device("cpu"), model_name="lstm")
    y_true, y_score = trainer.predict_lstm_scores(loader)
    assert len(y_true) == len(y_score)
    assert set(y_true).issubset({0.0, 1.0})


def test_train_lstm_saves_best_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(checkpoints.config, "PROJECT_ROOT", root)

    train_loader = _make_lstm_loader()
    val_loader = _make_lstm_loader()
    trainer = ModelTrainer(AudioLSTMDetector(), device=torch.device("cpu"), model_name="lstm")
    _ = trainer.train_lstm_detector(train_loader, val_loader, epochs=2, save_checkpoints=True)

    best = checkpoints.get_best_checkpoint("lstm")
    assert best is not None
    assert best.path.exists()
    assert best.epoch >= 1
