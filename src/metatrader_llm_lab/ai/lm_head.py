"""Lesson 09: project Transformer states into vocabulary logits and probabilities."""

import torch
from torch import nn


class LanguageModelHead(nn.Module):
    """Convert each Transformer vector into one score per vocabulary token."""

    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")

        self.projection = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return raw next-token scores (logits)."""
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [sequence, embedding_dim]")
        return self.projection(hidden_states)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        """Convert logits into a probability distribution."""
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def greedy_next_token(logits: torch.Tensor) -> int:
        """Choose the highest-scoring token from the final sequence position."""
        if logits.ndim != 2:
            raise ValueError("logits must have shape [sequence, vocab_size]")
        return int(torch.argmax(logits[-1], dim=-1).item())
