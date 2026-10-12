"""Lesson 15: compare sequential and vectorized inference on the same CPU."""

import argparse
import statistics
import time

import torch

from metatrader_llm_lab.ai.batching import make_token_batch
from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel
from metatrader_llm_lab.ai.vectorized import VectorizedMiniLanguageModel


def measure(fn, repetitions: int = 10, warmup: int = 3) -> float:
    """Return median wall-clock milliseconds per call, after warmup."""
    if repetitions < 1 or warmup < 0:
        raise ValueError("invalid benchmark repetition count")
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def run_benchmark(batch_size: int = 8, sequence_length: int = 16,
                  repetitions: int = 10) -> dict[str, float]:
    """Compare inference only, not training; model weights are independent."""
    if batch_size < 1 or sequence_length < 2:
        raise ValueError("batch_size >= 1 and sequence_length >= 2 required")
    torch.manual_seed(15)
    torch.set_num_threads(1)
    sequences = [torch.randint(1, 64, (sequence_length,), dtype=torch.long)
                 for _ in range(batch_size)]
    batch = make_token_batch(sequences)
    sequential = MiniLanguageModel(vocab_size=64).eval()
    vectorized = VectorizedMiniLanguageModel(vocab_size=64).eval()

    def sequential_call() -> None:
        for sequence in sequences:
            sequential(sequence)

    def vectorized_call() -> None:
        vectorized(batch.token_ids, batch.attention_mask)

    with torch.inference_mode():
        sequential_ms = measure(sequential_call, repetitions)
        vectorized_ms = measure(vectorized_call, repetitions)

    return {
        "sequential_ms": sequential_ms,
        "vectorized_ms": vectorized_ms,
        "sequential_samples_per_sec": batch_size * 1000 / sequential_ms,
        "vectorized_samples_per_sec": batch_size * 1000 / vectorized_ms,
        "speedup": sequential_ms / vectorized_ms,
        "sequential_parameter_bytes": sum(p.numel() * p.element_size()
                                          for p in sequential.parameters()),
        "vectorized_parameter_bytes": sum(p.numel() * p.element_size()
                                          for p in vectorized.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    results = run_benchmark(args.batch_size, args.sequence_length,
                            args.repetitions)
    for key, value in results.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
