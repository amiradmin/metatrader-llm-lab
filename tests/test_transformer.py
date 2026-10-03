import torch

from metatrader_llm_lab.ai.transformer import FeedForward, TransformerBlock


def test_feed_forward_preserves_token_shape() -> None:
    layer = FeedForward(embedding_dim=8, hidden_dim=32)
    output = layer(torch.randn(4, 8))

    assert output.shape == (4, 8)


def test_transformer_block_preserves_shape() -> None:
    block = TransformerBlock(embedding_dim=8, num_heads=2, hidden_dim=32)
    output, weights = block(torch.randn(4, 8))

    assert output.shape == (4, 8)
    assert weights.shape == (2, 4, 4)


def test_transformer_block_keeps_causal_attention() -> None:
    block = TransformerBlock(embedding_dim=8, num_heads=2, hidden_dim=32)
    _, weights = block(torch.randn(4, 8))

    assert weights[:, 0, 1:].eq(0).all()
    assert weights[:, 1, 2:].eq(0).all()
    assert weights[:, 2, 3:].eq(0).all()
