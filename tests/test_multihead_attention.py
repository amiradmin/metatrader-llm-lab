import pytest
import torch

from metatrader_llm_lab.ai.multihead_attention import MultiHeadCausalSelfAttention


def test_multihead_shapes() -> None:
    layer = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2)
    output, weights = layer(torch.randn(4, 8))

    assert output.shape == (4, 8)
    assert weights.shape == (2, 4, 4)


def test_each_head_attention_row_sums_to_one() -> None:
    layer = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2)
    _, weights = layer(torch.randn(4, 8))

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_each_head_masks_future_tokens() -> None:
    layer = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2)
    _, weights = layer(torch.randn(4, 8))

    assert weights[:, 0, 1:].eq(0).all()
    assert weights[:, 1, 2:].eq(0).all()
    assert weights[:, 2, 3:].eq(0).all()


def test_embedding_dimension_must_split_across_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadCausalSelfAttention(embedding_dim=7, num_heads=2)
