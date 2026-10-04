import torch

from metatrader_llm_lab.ai.lm_head import LanguageModelHead


def test_lm_head_projects_to_vocabulary() -> None:
    head = LanguageModelHead(embedding_dim=8, vocab_size=25)
    logits = head(torch.randn(4, 8))

    assert logits.shape == (4, 25)


def test_probabilities_sum_to_one() -> None:
    head = LanguageModelHead(embedding_dim=8, vocab_size=25)
    logits = head(torch.randn(4, 8))
    probabilities = head.probabilities(logits)

    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_greedy_next_token_uses_final_position() -> None:
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0],
            [0.1, 0.2, 4.0],
        ]
    )

    assert LanguageModelHead.greedy_next_token(logits) == 2
