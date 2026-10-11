import torch

from metatrader_llm_lab.ai.vectorized import (
    VectorizedMiniLanguageModel, vectorized_next_token_loss,
)


def test_vectorized_shapes_and_finite_loss() -> None:
    model = VectorizedMiniLanguageModel(vocab_size=16)
    sequences = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7])]
    loss = vectorized_next_token_loss(model, sequences)
    assert torch.isfinite(loss)
    logits = model(torch.tensor([[1, 2, 3], [4, 5, 0]]),
                   torch.tensor([[True, True, True], [True, True, False]]))
    assert logits.shape == (2, 3, 16)


def test_padding_does_not_change_real_logits() -> None:
    torch.manual_seed(14)
    model = VectorizedMiniLanguageModel(vocab_size=16)
    model.eval()
    with torch.no_grad():
        alone = model(torch.tensor([[1, 2]]), torch.tensor([[True, True]]))
        padded = model(torch.tensor([[1, 2, 0]]),
                       torch.tensor([[True, True, False]]))
    assert torch.allclose(alone[0], padded[0, :2], atol=1e-5)


def test_future_tokens_do_not_change_past_logits() -> None:
    torch.manual_seed(14)
    model = VectorizedMiniLanguageModel(vocab_size=16)
    model.eval()
    with torch.no_grad():
        a = model(torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.bool))
        b = model(torch.tensor([[1, 2, 9]]), torch.ones(1, 3, dtype=torch.bool))
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-5)
