import torch

from metatrader_llm_lab.ai.position import LearnedPositionEmbedding


def test_position_embedding_preserves_shape() -> None:
    layer = LearnedPositionEmbedding(max_sequence_length=64, embedding_dim=8)
    token_vectors = torch.zeros(36, 8)

    positioned = layer(token_vectors)

    assert positioned.shape == (36, 8)


def test_same_token_vector_changes_at_different_positions() -> None:
    layer = LearnedPositionEmbedding(max_sequence_length=8, embedding_dim=8)
    same_token = torch.ones(3, 8)

    positioned = layer(same_token)

    assert not torch.equal(positioned[0], positioned[1])
    assert not torch.equal(positioned[1], positioned[2])


def test_sequence_longer_than_limit_is_rejected() -> None:
    layer = LearnedPositionEmbedding(max_sequence_length=2, embedding_dim=8)

    try:
        layer(torch.zeros(3, 8))
    except ValueError as exc:
        assert "longer" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
