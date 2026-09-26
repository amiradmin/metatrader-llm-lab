"""Lesson 03: a tiny tokenizer for learning how text becomes token IDs."""

import re
from dataclasses import dataclass, field


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.%+-]+|[^\w\s]", re.UNICODE)


@dataclass
class SimpleTokenizer:
    """Small educational tokenizer with a vocabulary learned from supplied text."""

    token_to_id: dict[str, int] = field(default_factory=lambda: {"<UNK>": 0})
    id_to_token: dict[int, str] = field(default_factory=lambda: {0: "<UNK>"})

    def split(self, text: str) -> list[str]:
        """Split text into visible pieces before assigning integer IDs."""
        return _TOKEN_PATTERN.findall(text)

    def fit(self, texts: list[str]) -> None:
        """Add tokens from training text to the vocabulary."""
        for text in texts:
            for token in self.split(text):
                if token not in self.token_to_id:
                    token_id = len(self.token_to_id)
                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs; unseen pieces become <UNK>."""
        return [self.token_to_id.get(token, 0) for token in self.split(text)]

    def decode(self, token_ids: list[int]) -> list[str]:
        """Map token IDs back to their token strings for inspection."""
        return [self.id_to_token.get(token_id, "<UNK>") for token_id in token_ids]
