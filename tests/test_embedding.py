import torch

from metatrader_llm_lab.ai.embedding import TokenEmbedding


def test_embedding_shape() -> None:
    layer = TokenEmbedding(vocab_size=25, embedding_dim=8)
    token_ids = torch.tensor([1, 2, 3, 2], dtype=torch.long)

    vectors = layer(token_ids)

    assert vectors.shape == (4, 8)


def test_same_token_id_gets_same_vector() -> None:
    layer = TokenEmbedding(vocab_size=25, embedding_dim=8)
    token_ids = torch.tensor([2, 5, 2], dtype=torch.long)

    vectors = layer(token_ids)

    assert torch.equal(vectors[0], vectors[2])
    assert not torch.equal(vectors[0], vectors[1])
