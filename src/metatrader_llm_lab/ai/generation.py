"""Lesson 16: autoregressive generation from a causal language model."""

import torch

from metatrader_llm_lab.ai.mini_lm import MiniLanguageModel


def select_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
) -> int:
    """Choose a token from one vocabulary-logit vector."""
    if logits.ndim != 1:
        raise ValueError("logits must have shape [vocabulary]")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if temperature == 0:
        return int(torch.argmax(logits).item())
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probabilities, 1).item())


@torch.inference_mode()
def generate_tokens(
    model: MiniLanguageModel,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    eos_token_id: int | None = None,
) -> list[int]:
    """Return prompt plus generated tokens; stop at EOS or token budget."""
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    context_limit = model.position.position_embedding.num_embeddings
    if len(prompt_ids) > context_limit:
        raise ValueError("prompt exceeds model context length")

    was_training = model.training
    model.eval()
    try:
        generated = list(prompt_ids)
        device = next(model.parameters()).device
        for _ in range(max_new_tokens):
            context = torch.tensor(
                generated[-context_limit:], dtype=torch.long, device=device
            )
            logits = model(context)
            next_id = select_next_token(
                logits[-1], temperature=temperature
            )
            generated.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                break
        return generated
    finally:
        model.train(was_training)
