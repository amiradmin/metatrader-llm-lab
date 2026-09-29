import torch

from metatrader_llm_lab.ai.attention import CausalSelfAttention


def test_attention_shapes() -> None:
    layer = CausalSelfAttention(embedding_dim=8)
    token_vectors = torch.randn(4, 8)

    output, weights = layer(token_vectors)

    assert output.shape == (4, 8)
    assert weights.shape == (4, 4)


def test_each_attention_row_sums_to_one() -> None:
    layer = CausalSelfAttention(embedding_dim=8)
    _, weights = layer(torch.randn(4, 8))

    expected = torch.ones(4)
    assert torch.allclose(weights.sum(dim=-1), expected, atol=1e-6)


def test_future_tokens_are_masked() -> None:
    layer = CausalSelfAttention(embedding_dim=8)
    _, weights = layer(torch.randn(4, 8))

    assert weights[0, 1:].eq(0).all()
    assert weights[1, 2:].eq(0).all()
    assert weights[2, 3:].eq(0).all()
