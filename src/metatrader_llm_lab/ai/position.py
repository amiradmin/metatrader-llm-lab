"""Lesson 05: positional information for token embeddings."""

import torch
from torch import nn


class LearnedPositionEmbedding(nn.Module):
    """Add a trainable vector that identifies each token position."""

    def __init__(self, max_sequence_length: int, embedding_dim: int) -> None:
        super().__init__()
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        self.position_embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_vectors: torch.Tensor) -> torch.Tensor:
        """Add the matching position vector to every token vector."""
        if token_vectors.ndim != 2:
            raise ValueError("token_vectors must have shape [sequence, embedding_dim]")

        sequence_length = token_vectors.shape[0]
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError("sequence is longer than max_sequence_length")

        positions = torch.arange(sequence_length, device=token_vectors.device)
        return token_vectors + self.position_embedding(positions)
