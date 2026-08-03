"""PPO/GRPO clipping tests that expose positive and negative advantage behavior."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from nanopt.core.clipping import clipped_policy_loss, probability_ratio


def test_probability_ratio_is_one_for_unchanged_log_probabilities() -> None:
    logps = torch.tensor([[-1.0, -2.0]])

    assert torch.equal(probability_ratio(logps, logps), torch.ones_like(logps))


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected_loss", "is_clipped"),
    [
        (1.5, 1.0, -1.2, True),
        (0.5, 1.0, -0.5, False),
        (0.5, -1.0, 0.8, True),
        (1.5, -1.0, 1.5, False),
    ],
)
def test_clipping_depends_on_advantage_sign(
    ratio: float,
    advantage: float,
    expected_loss: float,
    is_clipped: bool,
) -> None:
    current = torch.tensor([[math.log(ratio)]])
    old = torch.zeros_like(current)

    result = clipped_policy_loss(
        current,
        old,
        torch.tensor([advantage]),
        torch.ones_like(current),
        clip_epsilon=0.2,
    )

    assert result.loss.item() == pytest.approx(expected_loss)
    assert result.clipped.item() is is_clipped
    assert result.clip_fraction.item() == float(is_clipped)


def test_token_and_sequence_normalization_differ_for_unequal_lengths() -> None:
    current = torch.zeros((2, 3))
    old = torch.zeros_like(current)
    advantages = torch.tensor([1.0, 3.0])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])

    token_mean = clipped_policy_loss(
        current,
        old,
        advantages,
        mask,
        clip_epsilon=0.2,
        normalization="token_mean",
    )
    sequence_mean = clipped_policy_loss(
        current,
        old,
        advantages,
        mask,
        clip_epsilon=0.2,
        normalization="sequence_mean",
    )

    assert token_mean.loss.item() == pytest.approx(-2.5)
    assert sequence_mean.loss.item() == pytest.approx(-2.0)


def test_clipped_policy_loss_preserves_gradients() -> None:
    current = torch.zeros((1, 2), requires_grad=True)
    result = clipped_policy_loss(
        current,
        torch.zeros_like(current),
        torch.tensor([1.0]),
        torch.ones_like(current),
        clip_epsilon=0.2,
    )

    result.loss.backward()

    assert current.grad is not None
    assert torch.equal(current.grad, torch.tensor([[-0.5, -0.5]]))


def test_clipped_policy_loss_accepts_token_level_advantages() -> None:
    current = torch.zeros((1, 2))
    result = clipped_policy_loss(
        current,
        current,
        torch.tensor([[1.0, 2.0]]),
        torch.ones_like(current),
        clip_epsilon=0.2,
    )

    assert result.loss.item() == pytest.approx(-1.5)


@pytest.mark.parametrize(
    ("current", "old", "kwargs", "message"),
    [
        (torch.ones(2), torch.ones(3), {}, "identical shapes"),
        (torch.empty(0), torch.empty(0), {}, "must not be empty"),
        (torch.ones(2, dtype=torch.long), torch.ones(2), {}, "floating point"),
        (torch.tensor([0.0, float("nan")]), torch.zeros(2), {}, "finite"),
        (torch.zeros(1), torch.zeros(1), {"max_abs_log_ratio": 0}, "must be positive"),
        (torch.tensor([100.0]), torch.zeros(1), {}, "exceeds diagnostic bound"),
    ],
)
def test_probability_ratio_rejects_invalid_contracts(
    current: torch.Tensor,
    old: torch.Tensor,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        probability_ratio(current, old, **kwargs)


def test_probability_ratio_rejects_device_mismatch() -> None:
    with pytest.raises(ValueError, match="same device"):
        probability_ratio(torch.ones(1), torch.ones(1, device="meta"))


@pytest.mark.parametrize(
    ("advantages", "mask", "kwargs", "message"),
    [
        (torch.ones(1), torch.ones((1, 3)), {}, "action_mask must match"),
        (torch.ones(1), torch.ones(1), {}, "at least batch and token"),
        (torch.ones((1, 2, 1)), torch.ones((1, 2)), {}, "advantages must match"),
        (torch.ones(1, dtype=torch.long), torch.ones((1, 2)), {}, "floating point"),
        (torch.tensor([float("inf")]), torch.ones((1, 2)), {}, "finite"),
        (torch.ones(1), torch.ones((1, 2)), {"clip_epsilon": 0}, "between 0 and 1"),
        (torch.ones(1), torch.ones((1, 2)), {"normalization": "bad"}, "unknown"),
    ],
)
def test_clipped_loss_rejects_invalid_contracts(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    current = torch.zeros((1, 2)) if mask.ndim == 2 else torch.zeros(1)
    old = torch.zeros_like(current)
    options: dict[str, Any] = {"clip_epsilon": 0.2, **kwargs}
    with pytest.raises((TypeError, ValueError), match=message):
        clipped_policy_loss(current, old, advantages, mask, **options)


def test_clipped_loss_rejects_device_mismatch_and_empty_sequence() -> None:
    current = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="share a device"):
        clipped_policy_loss(
            current,
            current,
            torch.ones(1, device="meta"),
            torch.ones_like(current),
            clip_epsilon=0.2,
        )
    with pytest.raises(ValueError, match="zero active values"):
        clipped_policy_loss(
            current,
            current,
            torch.ones(1),
            torch.zeros_like(current),
            clip_epsilon=0.2,
        )
