import torch

from metatrader_llm_lab.ai.batching import make_token_batch


def test_batch_shape_uses_longest_sequence() -> None:
    batch = make_token_batch(
        [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5], dtype=torch.long),
        ]
    )

    assert batch.token_ids.shape == (2, 3)
    assert batch.attention_mask.shape == (2, 3)


def test_short_sequence_is_padded() -> None:
    batch = make_token_batch(
        [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5], dtype=torch.long),
        ],
        pad_token_id=0,
    )

    assert batch.token_ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch.attention_mask.tolist() == [
        [True, True, True],
        [True, True, False],
    ]


def test_custom_padding_token_is_supported() -> None:
    batch = make_token_batch(
        [
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2, 3], dtype=torch.long),
        ],
        pad_token_id=9,
    )

    assert batch.token_ids.tolist() == [[1, 9], [2, 3]]
