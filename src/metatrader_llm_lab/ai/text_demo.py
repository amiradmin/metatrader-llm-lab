"""Lesson 17: train a tiny language model on a toy market-text corpus.

Run: uv run python -m metatrader_llm_lab.ai.text_demo
"""

import argparse

import torch

from metatrader_llm_lab.ai.generation import generate_tokens
from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel, next_token_loss
from metatrader_llm_lab.ai.tokenizer import SimpleTokenizer


CORPUS = [
    "market trend is up",
    "market trend is down",
    "market trend is range",
    "market volume is high",
    "market volume is low",
]


def run_demo(steps: int = 120, learning_rate: float = 0.01) -> dict:
    """Train on a tiny repeated corpus, then generate from a familiar prefix."""
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")

    torch.manual_seed(17)
    tokenizer = SimpleTokenizer()
    tokenizer.fit(CORPUS)
    sequences = [
        torch.tensor(tokenizer.encode(line), dtype=torch.long)
        for line in CORPUS
    ]
    model = MiniLanguageModel(
        vocab_size=len(tokenizer.token_to_id),
        embedding_dim=16,
        num_heads=2,
        hidden_dim=32,
        max_sequence_length=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def mean_loss() -> float:
        model.eval()
        with torch.no_grad():
            return sum(next_token_loss(model, s).item() for s in sequences) / len(sequences)

    initial_loss = mean_loss()
    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        loss = next_token_loss(model, sequences[step % len(sequences)])
        loss.backward()
        optimizer.step()

    final_loss = mean_loss()
    prompt = "market trend is"
    prompt_ids = tokenizer.encode(prompt)
    output_ids = generate_tokens(model, prompt_ids, max_new_tokens=1)
    generated_token = tokenizer.decode(output_ids[len(prompt_ids):])[0]
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "prompt": prompt,
        "generated_token": generated_token,
        "vocabulary_size": len(tokenizer.token_to_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    result = run_demo(steps=args.steps)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
