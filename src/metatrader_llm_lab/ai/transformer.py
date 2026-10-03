"""Lesson 08: a small pre-norm Transformer block."""

import torch
from torch import nn

from metatrader_llm_lab.ai.multihead_attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    """Process each token independently with a small nonlinear network."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, token_vectors: torch.Tensor) -> torch.Tensor:
        return self.network(token_vectors)


class TransformerBlock(nn.Module):
    """Combine attention, feed-forward layers, residual paths and normalization."""

    def __init__(
        self,
        embedding_dim: int = 8,
        num_heads: int = 2,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)

    def forward(
        self,
        token_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_input = self.norm1(token_vectors)
        attention_output, weights = self.attention(attention_input)
        token_vectors = token_vectors + attention_output

        feed_forward_input = self.norm2(token_vectors)
        token_vectors = token_vectors + self.feed_forward(feed_forward_input)

        return token_vectors, weights
