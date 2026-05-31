from __future__ import annotations

import torch
import torch.nn as nn


class AudioLSTMDetector(nn.Module):
    """
    Sequence-to-sequence detector for frame-level binary classification.
    Input: [B, T, 128]
    Output logits: [B, T, 1]
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.lstm(x)
        return self.head(features)
