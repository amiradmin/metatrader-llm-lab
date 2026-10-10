import torch

from metatrader_llm_lab.ai.batched_training import (
    batched_next_token_loss,
    train_batch_step,
)
from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel, next_token_loss


def test_batch_loss_matches_weighted_individual_losses() -> None:
    torch.manual_seed(13)
    model = MiniLanguageModel(vocab_size=12)
    sequences = [
        torch.tensor([1, 2, 3, 4], dtype=torch.long),
        torch.tensor([5, 6, 7], dtype=torch.long),
    ]
    actual = batched_next_token_loss(model, sequences)
    expected = (
        next_token_loss(model, sequences[0]) * 3
        + next_token_loss(model, sequences[1]) * 2
    ) / 5
    assert torch.allclose(actual, expected, atol=1e-6)


def test_batch_training_updates_weights() -> None:
    torch.manual_seed(13)
    model = MiniLanguageModel(vocab_size=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.token_embedding.embedding.weight.detach().clone()
    loss = train_batch_step(
        model,
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])],
        optimizer,
    )
    assert loss > 0
    assert not torch.equal(before, model.token_embedding.embedding.weight.detach())
