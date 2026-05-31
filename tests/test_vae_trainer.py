import matplotlib
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import checkpoints
from src.models.trainer import ModelTrainer
from src.models.vae import VAESynthesizer
from src.visualization.visualize import mel_to_audio, plot_spectrogram_pair

matplotlib.use("Agg")


def _make_vae_loader(n_samples: int = 8) -> DataLoader:
    x = torch.randn(n_samples, 1, 128, 128)
    y = torch.zeros(n_samples, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_train_vae_one_epoch_and_history():
    train_loader = _make_vae_loader()
    val_loader = _make_vae_loader()

    trainer = ModelTrainer(VAESynthesizer(), device=torch.device("cpu"), model_name="vae")
    history = trainer.train_vae(train_loader, val_loader, epochs=1)

    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert len(history["train_recon"]) == 1
    assert len(history["train_kl"]) == 1
    assert len(history["beta_kl"]) == 1


def test_train_vae_saves_best_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(checkpoints.config, "PROJECT_ROOT", root)

    train_loader = _make_vae_loader()
    val_loader = _make_vae_loader()

    trainer = ModelTrainer(VAESynthesizer(), device=torch.device("cpu"), model_name="vae")
    _ = trainer.train_vae(train_loader, val_loader, epochs=2, save_checkpoints=True)
    best = checkpoints.get_best_checkpoint("vae")
    assert best is not None
    assert best.path.exists()


def test_train_vae_kl_warmup_increases_beta():
    train_loader = _make_vae_loader()
    val_loader = _make_vae_loader()
    trainer = ModelTrainer(VAESynthesizer(), device=torch.device("cpu"), model_name="vae")
    history = trainer.train_vae(
        train_loader, val_loader, epochs=3, beta_kl=1.0, kl_warmup_epochs=3, save_checkpoints=False
    )
    assert history["beta_kl"][0] < history["beta_kl"][-1]
    assert history["beta_kl"][-1] == 1.0


def test_vae_visualization_helpers_smoke():
    original = torch.randn(1, 128, 128)
    recon = torch.randn(1, 128, 128)
    plot_spectrogram_pair(original, recon)

    waveform = mel_to_audio(original, input_scale="log")
    assert waveform.dim() == 2
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0
