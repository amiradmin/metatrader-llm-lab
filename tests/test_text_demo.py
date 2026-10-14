import pytest

from metatrader_llm_lab.ai.text_demo import run_demo


def test_toy_corpus_training_reduces_loss() -> None:
    result = run_demo(steps=80)
    assert result["final_loss"] < result["initial_loss"]
    assert result["generated_token"] in {"up", "down", "range"}


def test_invalid_training_steps_are_rejected() -> None:
    with pytest.raises(ValueError):
        run_demo(steps=0)
