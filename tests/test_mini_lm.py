import torch

from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel, next_token_loss


def test_mini_lm_output_shape() -> None:
    model = MiniLanguageModel(vocab_size=25)
    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    logits = model(token_ids)

    assert logits.shape == (4, 25)


def test_next_token_loss_is_scalar_and_finite() -> None:
    model = MiniLanguageModel(vocab_size=25)
    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    loss = next_token_loss(model, token_ids)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_training_step_updates_parameters() -> None:
    torch.manual_seed(7)
    model = MiniLanguageModel(vocab_size=25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    token_ids = torch.tensor([1, 2, 3, 4, 2, 3, 4], dtype=torch.long)

    before = model.token_embedding.embedding.weight.detach().clone()

    optimizer.zero_grad()
    loss = next_token_loss(model, token_ids)
    loss.backward()
    optimizer.step()

    after = model.token_embedding.embedding.weight.detach()

    assert not torch.equal(before, after)
