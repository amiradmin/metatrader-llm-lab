"""Lesson 04: a tiny trainable embedding table built with PyTorch."""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Map token IDs to trainable dense vectors."""

    def __init__(self, vocab_size: int, embedding_dim: int = 8) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return one embedding vector for every token ID."""
        return self.embedding(token_ids)
