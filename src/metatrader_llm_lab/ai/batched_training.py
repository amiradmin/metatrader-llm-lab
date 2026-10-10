"""Lesson 13: train a padded mini-batch without mixing independent sequences."""

import torch
from torch import nn

from metatrader_llm_lab.ai.batching import make_token_batch
from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel


def batched_next_token_loss(
    model: MiniLanguageModel,
    sequences: list[torch.Tensor],
) -> torch.Tensor:
    """Average next-token loss over valid targets, excluding padded positions.

    The current model handles one sequence at a time. This educational bridge
    batches the loss while keeping each sequence's attention independent.
    """
    if not sequences or any(s.ndim != 1 or s.numel() < 2 for s in sequences):
        raise ValueError("each sequence must contain at least two tokens")

    batch = make_token_batch(sequences)
    losses = []
    counts = []
    for ids, mask in zip(batch.token_ids, batch.attention_mask):
        length = int(mask.sum().item())
        logits = model(ids[: length - 1])
        targets = ids[1:length]
        losses.append(nn.functional.cross_entropy(logits, targets, reduction="sum"))
        counts.append(length - 1)
    return torch.stack(losses).sum() / sum(counts)


def train_batch_step(
    model: MiniLanguageModel,
    sequences: list[torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Perform one optimizer update for a mini-batch."""
    model.train()
    optimizer.zero_grad()
    loss = batched_next_token_loss(model, sequences)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())
