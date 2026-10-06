import torch

from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel
from metatrader_llm_lab.ai.training import average_loss, train_epochs


def test_average_loss_is_finite() -> None:
    model = MiniLanguageModel(vocab_size=10)
    sequences = [torch.tensor([1, 2, 3, 4], dtype=torch.long)]

    loss = average_loss(model, sequences)

    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))


def test_train_epochs_returns_metrics_for_every_epoch() -> None:
    torch.manual_seed(7)
    model = MiniLanguageModel(vocab_size=10)
    train = [torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)]
    validation = [torch.tensor([1, 2, 3, 4, 6], dtype=torch.long)]

    history = train_epochs(model, train, validation, epochs=3)

    assert len(history) == 3
    assert [row.epoch for row in history] == [1, 2, 3]
    assert all(row.train_loss > 0 for row in history)
    assert all(row.validation_loss > 0 for row in history)


def test_training_can_reduce_loss_on_repeated_pattern() -> None:
    torch.manual_seed(7)
    model = MiniLanguageModel(vocab_size=10)
    train = [torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.long)]
    validation = [torch.tensor([1, 2, 1, 2], dtype=torch.long)]

    before = average_loss(model, train)
    train_epochs(model, train, validation, epochs=8, learning_rate=0.02)
    after = average_loss(model, train)

    assert after < before
