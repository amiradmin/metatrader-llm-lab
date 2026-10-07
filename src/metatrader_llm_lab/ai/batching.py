"""Lesson 12: mini-batches and padding for variable-length token sequences."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenBatch:
    """A padded batch plus a mask that marks real tokens."""

    token_ids: torch.Tensor
    attention_mask: torch.Tensor


def make_token_batch(
    sequences: list[torch.Tensor],
    pad_token_id: int = 0,
) -> TokenBatch:
    """Pad one-dimensional token sequences to a shared length."""
    if not sequences:
        raise ValueError("sequences must not be empty")
    if any(sequence.ndim != 1 for sequence in sequences):
        raise ValueError("each sequence must have shape [sequence]")
    if any(sequence.numel() == 0 for sequence in sequences):
        raise ValueError("sequences must not be empty")

    max_length = max(sequence.numel() for sequence in sequences)
    token_ids = torch.full(
        (len(sequences), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.bool,
    )

    for row, sequence in enumerate(sequences):
        length = sequence.numel()
        token_ids[row, :length] = sequence
        attention_mask[row, :length] = True

    return TokenBatch(token_ids=token_ids, attention_mask=attention_mask)
