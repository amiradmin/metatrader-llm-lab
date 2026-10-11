"""Lesson 14: genuinely vectorized causal Transformer training with padding."""

import torch
from torch import nn

from metatrader_llm_lab.ai.batching import make_token_batch
from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel


class VectorizedMiniLanguageModel(nn.Module):
    """Batch-first mini language model with independent causal attention."""

    def __init__(self, vocab_size: int, embedding_dim: int = 8,
                 num_heads: int = 2, hidden_dim: int = 32,
                 max_sequence_length: int = 128) -> None:
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, embedding_dim))
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or valid_mask.shape != token_ids.shape:
            raise ValueError("expected matching [batch, sequence] tensors")
        batch, length = token_ids.shape
        if not bool(valid_mask[:, 0].all()):
            raise ValueError("every sequence must start with a real token")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal = torch.triu(torch.ones(length, length, device=x.device,
                                       dtype=torch.bool), diagonal=1)
        q = self.norm1(x)
        attended, _ = self.attention(q, q, q, attn_mask=causal,
                                     key_padding_mask=~valid_mask,
                                     need_weights=False)
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return self.head(self.final_norm(x))


def vectorized_next_token_loss(
    model: VectorizedMiniLanguageModel,
    sequences: list[torch.Tensor],
) -> torch.Tensor:
    """Compute cross-entropy only for real next-token targets."""
    if not sequences or any(s.ndim != 1 or s.numel() < 2 for s in sequences):
        raise ValueError("each sequence must contain at least two tokens")
    batch = make_token_batch(sequences)
    inputs = batch.token_ids[:, :-1]
    input_mask = batch.attention_mask[:, :-1]
    targets = batch.token_ids[:, 1:]
    target_mask = batch.attention_mask[:, 1:]
    logits = model(inputs, input_mask)
    return nn.functional.cross_entropy(logits[target_mask], targets[target_mask])
