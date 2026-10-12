import pytest
import torch

from metatrader_llm_lab.ai.generation import generate_tokens, select_next_token
from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel


def test_greedy_selects_highest_logit() -> None:
    assert select_next_token(torch.tensor([0.2, 3.0, 1.0])) == 1


def test_generation_appends_requested_number_of_tokens() -> None:
    torch.manual_seed(16)
    model = MiniLanguageModel(vocab_size=12)
    result = generate_tokens(model, [1, 2, 3], max_new_tokens=4)
    assert len(result) == 7
    assert result[:3] == [1, 2, 3]
    assert all(0 <= token < 12 for token in result)


def test_generation_preserves_model_training_mode() -> None:
    model = MiniLanguageModel(vocab_size=12)
    model.train()
    generate_tokens(model, [1, 2], max_new_tokens=1)
    assert model.training


def test_invalid_temperature_is_rejected() -> None:
    with pytest.raises(ValueError):
        select_next_token(torch.tensor([1.0, 2.0]), temperature=-1.0)


def test_long_prompt_is_rejected() -> None:
    model = MiniLanguageModel(vocab_size=12, max_sequence_length=4)
    with pytest.raises(ValueError):
        generate_tokens(model, [1, 2, 3, 4, 5])
