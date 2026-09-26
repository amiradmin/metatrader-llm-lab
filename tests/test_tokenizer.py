from metatrader_llm_lab.ai.tokenizer import SimpleTokenizer


def test_tokenizer_round_trip_known_tokens() -> None:
    tokenizer = SimpleTokenizer()
    training_text = "XAUUSD short-term return is negative"
    tokenizer.fit([training_text])

    token_ids = tokenizer.encode(training_text)

    assert len(token_ids) == len(tokenizer.split(training_text))
    assert 0 not in token_ids
    assert tokenizer.decode(token_ids) == tokenizer.split(training_text)


def test_unknown_token_uses_unk_id() -> None:
    tokenizer = SimpleTokenizer()
    tokenizer.fit(["XAUUSD return negative"])

    assert tokenizer.encode("unseen") == [0]
    assert tokenizer.decode([0]) == ["<UNK>"]
