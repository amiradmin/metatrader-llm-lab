"""Lesson 07: multi-head causal self-attention."""

import torch
from torch import nn


class MultiHeadCausalSelfAttention(nn.Module):
    """Run several causal attention heads in parallel and combine them."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            bias=False,
            batch_first=True,
        )

    def forward(
        self,
        token_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return combined vectors and per-head causal attention weights."""
        if token_vectors.ndim != 2:
            raise ValueError("token_vectors must have shape [sequence, embedding_dim]")

        sequence_length = token_vectors.shape[0]
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=token_vectors.device,
            ),
            diagonal=1,
        )

        batch = token_vectors.unsqueeze(0)
        output, weights = self.attention(
            batch,
            batch,
            batch,
            attn_mask=causal_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        return output.squeeze(0), weights.squeeze(0)
