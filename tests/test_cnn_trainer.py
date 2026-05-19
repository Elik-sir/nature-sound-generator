import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import CNNClassifier
from src.models.trainer import ModelTrainer


def test_train_classifier_one_epoch():
    x = torch.randn(16, 1, 128, 128)
    y = torch.randint(0, 50, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    trainer = ModelTrainer(CNNClassifier(), device=torch.device("cpu"))
    history = trainer.train_classifier(loader, loader, epochs=1)

    assert len(history["train_loss"]) == 1
    assert len(history["val_acc"]) == 1
