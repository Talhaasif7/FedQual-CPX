"""
Recurrent Neural Network models for Shakespeare next-character prediction in FedQual-CPX.

Per Section 27.3:
    Architecture:
        Embedding(vocab_size, embedding_dim)
        LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        Linear(hidden_dim, vocab_size)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ShakespeareLSTM(nn.Module):
    """Character-level LSTM model for Shakespeare federated language modeling."""

    def __init__(
        self,
        vocab_size: int = 90,
        embedding_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_sequences: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        # x: [B, seq_len]
        embeds = self.embedding(x)  # [B, seq_len, embed_dim]
        out, hidden = self.lstm(embeds, hidden)  # out: [B, seq_len, hidden_dim]
        logits = self.fc(out)  # [B, seq_len, vocab_size]
        if return_sequences:
            return logits, hidden
        return logits[:, -1, :]


