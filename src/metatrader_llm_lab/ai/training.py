"""Lesson 11: train for multiple epochs and validate on unseen sequences."""

from dataclasses import dataclass

import torch

from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel, next_token_loss


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float


def average_loss(
    model: MiniLanguageModel,
    sequences: list[torch.Tensor],
) -> float:
    """Measure average next-token loss without updating model weights."""
    if not sequences:
        raise ValueError("sequences must not be empty")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        losses = [next_token_loss(model, sequence).item() for sequence in sequences]
    if was_training:
        model.train()

    return sum(losses) / len(losses)


def train_epochs(
    model: MiniLanguageModel,
    train_sequences: list[torch.Tensor],
    validation_sequences: list[torch.Tensor],
    epochs: int = 5,
    learning_rate: float = 0.01,
) -> list[EpochMetrics]:
    """Train repeatedly and record train/validation loss after every epoch."""
    if not train_sequences:
        raise ValueError("train_sequences must not be empty")
    if not validation_sequences:
        raise ValueError("validation_sequences must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for sequence in train_sequences:
            optimizer.zero_grad()
            loss = next_token_loss(model, sequence)
            loss.backward()
            optimizer.step()

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=average_loss(model, train_sequences),
                validation_loss=average_loss(model, validation_sequences),
            )
        )

    return history
