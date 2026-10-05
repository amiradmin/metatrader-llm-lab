"""Lesson 10: train the mini language model with next-token prediction."""

import torch
from torch import nn

from metatrader_llm_lab.ai.embedding import TokenEmbedding
from metatrader_llm_lab.ai.lm_head import LanguageModelHead
from metatrader_llm_lab.ai.position import LearnedPositionEmbedding
from metatrader_llm_lab.ai.transformer import TransformerBlock


class MiniLanguageModel(nn.Module):
    """A tiny causal language model assembled from the lessons so far."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 8,
        num_heads: int = 2,
        hidden_dim: int = 32,
        max_sequence_length: int = 128,
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position = LearnedPositionEmbedding(max_sequence_length, embedding_dim)
        self.transformer = TransformerBlock(embedding_dim, num_heads, hidden_dim)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = LanguageModelHead(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.token_embedding(token_ids)
        vectors = self.position(vectors)
        vectors, _ = self.transformer(vectors)
        vectors = self.final_norm(vectors)
        return self.lm_head(vectors)


def next_token_loss(model: MiniLanguageModel, token_ids: torch.Tensor) -> torch.Tensor:
    """Train position t to predict the real token at position t+1."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [sequence]")
    if token_ids.numel() < 2:
        raise ValueError("at least two token IDs are required")

    inputs = token_ids[:-1]
    targets = token_ids[1:]

    logits = model(inputs)
    return nn.functional.cross_entropy(logits, targets)
