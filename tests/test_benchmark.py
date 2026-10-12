import pytest

from metatrader_llm_lab.ai.benchmark import measure, run_benchmark


def test_measure_returns_nonnegative_time() -> None:
    assert measure(lambda: sum(range(10)), repetitions=2, warmup=0) >= 0


def test_benchmark_returns_positive_metrics() -> None:
    result = run_benchmark(batch_size=2, sequence_length=4, repetitions=2)
    assert result["sequential_ms"] > 0
    assert result["vectorized_ms"] > 0
    assert result["speedup"] > 0
    assert result["sequential_parameter_bytes"] > 0
    assert result["vectorized_parameter_bytes"] > 0


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_benchmark(batch_size=0)
