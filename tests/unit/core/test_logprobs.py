"""Causal log-probability fixtures with probabilities chosen for hand calculation."""

from __future__ import annotations

import math

import pytest
import torch

from nanopt.core.logprobs import causal_token_logps, completion_sequence_logps


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.tensor(
        [
            [
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.2, 0.3, 0.5],
                [0.4, 0.4, 0.2],
            ]
        ]
    )
    input_ids = torch.tensor([[2, 1, 0, 2]])
    return probabilities.log(), input_ids


def test_causal_token_logps_applies_the_one_token_shift() -> None:
    logits, input_ids = _fixture()

    result = causal_token_logps(logits, input_ids)

    # logits[0] predicts ID 1, logits[1] predicts ID 0, logits[2] predicts ID 2.
    expected = torch.tensor([[math.log(0.2), math.log(0.6), math.log(0.5)]])
    assert result.dtype == torch.float32
    assert torch.allclose(result, expected)


def test_completion_sequence_logps_sums_only_completion_positions() -> None:
    logits, input_ids = _fixture()
    action_mask = torch.tensor([[0, 0, 1, 1]])

    result = completion_sequence_logps(logits, input_ids, action_mask)

    # Completion token ID 0 has probability 0.6 and token ID 2 has probability 0.5.
    assert result.item() == pytest.approx(math.log(0.6) + math.log(0.5))


def test_completion_logps_are_invariant_to_masked_prompt_logits() -> None:
    logits, input_ids = _fixture()
    changed = logits.clone()
    changed[:, 0, :] = torch.tensor([100.0, -100.0, 0.0])
    action_mask = torch.tensor([[0, 0, 1, 1]])

    original = completion_sequence_logps(logits, input_ids, action_mask)
    modified = completion_sequence_logps(changed, input_ids, action_mask)

    assert torch.equal(original, modified)


def test_causal_token_logps_compute_in_fp32_and_preserve_gradients() -> None:
    logits, input_ids = _fixture()
    low_precision_logits = logits.to(torch.bfloat16).requires_grad_()

    result = causal_token_logps(low_precision_logits, input_ids)
    result.sum().backward()

    assert result.dtype == torch.float32
    assert low_precision_logits.grad is not None
    assert torch.isfinite(low_precision_logits.grad).all()


@pytest.mark.parametrize(
    ("logits", "input_ids", "message"),
    [
        (torch.ones((2, 3)), torch.ones((2, 3), dtype=torch.long), "logits must have shape"),
        (torch.ones((1, 2, 3)), torch.ones(2, dtype=torch.long), "input_ids must have shape"),
        (torch.ones((1, 3, 4)), torch.ones((1, 2), dtype=torch.long), "must match input_ids"),
        (torch.ones((1, 1, 4)), torch.zeros((1, 1), dtype=torch.long), "at least 2"),
        (torch.ones((1, 2, 0)), torch.zeros((1, 2), dtype=torch.long), "must not be empty"),
        (
            torch.ones((1, 2, 3), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.long),
            "floating-point dtype",
        ),
        (torch.ones((1, 2, 3)), torch.zeros((1, 2)), "integer dtype"),
        (torch.ones((1, 2, 3)), torch.tensor([[0, 3]]), "between 0 and 2"),
    ],
)
def test_causal_token_logps_reject_invalid_contracts(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        causal_token_logps(logits, input_ids)


def test_completion_sequence_logps_rejects_an_empty_completion() -> None:
    logits, input_ids = _fixture()

    with pytest.raises(ValueError, match="at least one predicted completion token"):
        completion_sequence_logps(logits, input_ids, torch.zeros_like(input_ids))


def test_causal_token_logps_rejects_tensors_on_different_devices() -> None:
    with pytest.raises(ValueError, match="same device"):
        causal_token_logps(
            torch.ones((1, 2, 3)),
            torch.zeros((1, 2), dtype=torch.long, device="meta"),
        )


def test_completion_sequence_logps_rejects_mask_shape_and_device() -> None:
    logits, input_ids = _fixture()
    with pytest.raises(ValueError, match="must match input_ids"):
        completion_sequence_logps(logits, input_ids, torch.ones((1, 3)))
    with pytest.raises(ValueError, match="same device"):
        completion_sequence_logps(
            logits,
            input_ids,
            torch.ones(input_ids.shape, device="meta"),
        )
