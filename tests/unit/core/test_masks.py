"""Hand-checkable examples for token-coordinate and causal-coordinate masks."""

from __future__ import annotations

import pytest
import torch

from nanopt.core.masks import causal_action_mask, completion_action_mask


def test_completion_mask_excludes_prompt_padding_and_post_terminal_tokens() -> None:
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ]
    )
    completion_starts = torch.tensor([2, 1])
    terminal_mask = torch.tensor(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )

    result = completion_action_mask(
        attention_mask,
        completion_starts,
        terminal_mask=terminal_mask,
        include_terminal=True,
    )

    # Row 0: prompt=[0,1], completion=[2,3,4], padding=[5].
    # Row 1: prompt=[0], completion=[1,2], post-EOS=[3], padding=[4,5].
    expected = torch.tensor(
        [
            [False, False, True, True, True, False],
            [False, True, True, False, False, False],
        ]
    )
    assert torch.equal(result, expected)


def test_completion_mask_can_exclude_the_terminal_token() -> None:
    result = completion_action_mask(
        torch.ones((1, 5), dtype=torch.bool),
        torch.tensor([2]),
        terminal_mask=torch.tensor([[0, 0, 0, 1, 0]]),
        include_terminal=False,
    )

    assert torch.equal(result, torch.tensor([[False, False, True, False, False]]))


def test_completion_mask_allows_an_explicit_empty_completion() -> None:
    result = completion_action_mask(torch.ones((1, 3)), torch.tensor([3]))

    assert torch.equal(result, torch.zeros((1, 3), dtype=torch.bool))


def test_causal_action_mask_drops_the_unpredicted_first_token() -> None:
    action_mask = torch.tensor([[1, 0, 1, 1]])

    result = causal_action_mask(action_mask)

    assert torch.equal(result, torch.tensor([[False, True, True]]))


@pytest.mark.parametrize(
    ("attention_mask", "starts", "message"),
    [
        (torch.ones(3), torch.tensor([0]), "attention_mask must have shape"),
        (torch.ones((0, 3)), torch.empty(0, dtype=torch.long), "non-empty"),
        (torch.ones((2, 3)), torch.tensor([0]), "completion_starts must have shape"),
        (torch.ones((1, 3)), torch.tensor([1.0]), "integer dtype"),
        (torch.ones((1, 3)), torch.tensor([-1]), "between 0 and 3"),
        (torch.tensor([[1, 2, 1]]), torch.tensor([1]), "only 0 and 1"),
        (torch.tensor([[1.0, float("nan"), 0.0]]), torch.tensor([1]), "finite"),
        (torch.ones((1, 3), dtype=torch.complex64), torch.tensor([1]), "numeric dtype"),
    ],
)
def test_completion_mask_rejects_ambiguous_inputs(
    attention_mask: torch.Tensor,
    starts: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        completion_action_mask(attention_mask, starts)


def test_completion_mask_rejects_a_terminal_on_padding() -> None:
    with pytest.raises(ValueError, match="cannot mark padded"):
        completion_action_mask(
            torch.tensor([[1, 1, 0]]),
            torch.tensor([1]),
            terminal_mask=torch.tensor([[0, 0, 1]]),
        )


def test_completion_mask_rejects_device_and_terminal_contract_errors() -> None:
    attention_mask = torch.ones((1, 3))
    starts = torch.tensor([1])

    with pytest.raises(ValueError, match="completion_starts and attention_mask"):
        completion_action_mask(
            attention_mask,
            torch.ones((1,), dtype=torch.long, device="meta"),
        )
    with pytest.raises(ValueError, match="terminal_mask must match"):
        completion_action_mask(attention_mask, starts, terminal_mask=torch.ones((1, 2)))
    with pytest.raises(ValueError, match="terminal_mask and attention_mask"):
        completion_action_mask(
            attention_mask,
            starts,
            terminal_mask=torch.ones((1, 3), device="meta"),
        )
    with pytest.raises(ValueError, match="terminal_mask must contain only 0 and 1"):
        completion_action_mask(
            attention_mask,
            starts,
            terminal_mask=torch.tensor([[0, 2, 0]]),
        )


def test_causal_action_mask_rejects_rank_length_and_binary_errors() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        causal_action_mask(torch.ones(3))
    with pytest.raises(ValueError, match="at least 2"):
        causal_action_mask(torch.ones((1, 1)))
    with pytest.raises(ValueError, match="only 0 and 1"):
        causal_action_mask(torch.tensor([[0, 2]]))
