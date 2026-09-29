"""Lesson 06: single-head causal self-attention for learning Q, K and V."""

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Let each token attend to itself and earlier tokens."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        token_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attended vectors and the attention-weight matrix."""
        if token_vectors.ndim != 2:
            raise ValueError("token_vectors must have shape [sequence, embedding_dim]")

        q = self.query(token_vectors)
        k = self.key(token_vectors)
        v = self.value(token_vectors)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.embedding_dim)

        sequence_length = token_vectors.shape[0]
        future_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=token_vectors.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(future_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        return output, weights
