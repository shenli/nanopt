"""DPO examples whose margins and logistic losses can be calculated by hand."""

from __future__ import annotations

import math

import pytest
import torch

from nanopt.core.dpo import dpo_loss, preference_margin


def test_matching_policy_and_reference_margins_have_log_two_loss() -> None:
    result = dpo_loss(
        torch.tensor([-2.0]),
        torch.tensor([-3.0]),
        torch.tensor([-4.0]),
        torch.tensor([-5.0]),
        beta=0.1,
    )

    assert result.policy_margin.item() == 1.0
    assert result.reference_margin.item() == 1.0
    assert result.implicit_reward_margin.item() == 0.0
    assert result.loss.item() == pytest.approx(math.log(2.0))


def test_increasing_chosen_policy_margin_lowers_dpo_loss() -> None:
    reference_chosen = torch.tensor([-2.0])
    reference_rejected = torch.tensor([-3.0])
    baseline = dpo_loss(
        torch.tensor([-2.0]),
        torch.tensor([-3.0]),
        reference_chosen,
        reference_rejected,
        beta=1.0,
    )
    improved = dpo_loss(
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        reference_chosen,
        reference_rejected,
        beta=1.0,
    )

    assert improved.loss < baseline.loss
    assert improved.implicit_reward_margin.item() == 1.0


def test_beta_scales_the_implicit_reward_margin() -> None:
    inputs = (
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        torch.tensor([-2.0]),
        torch.tensor([-3.0]),
    )

    low_beta = dpo_loss(*inputs, beta=0.1)
    high_beta = dpo_loss(*inputs, beta=1.0)

    assert high_beta.implicit_reward_margin.item() == pytest.approx(
        10 * low_beta.implicit_reward_margin.item()
    )
    assert high_beta.loss < low_beta.loss


def test_dpo_uses_fp32_and_preserves_policy_gradients() -> None:
    chosen = torch.tensor([-1.0], dtype=torch.bfloat16, requires_grad=True)
    rejected = torch.tensor([-2.0], dtype=torch.bfloat16, requires_grad=True)

    result = dpo_loss(chosen, rejected, torch.tensor([-1.0]), torch.tensor([-2.0]), beta=0.1)
    result.loss.backward()

    assert result.loss.dtype == torch.float32
    assert chosen.grad is not None
    assert rejected.grad is not None
    assert chosen.grad.item() < 0
    assert rejected.grad.item() > 0


@pytest.mark.parametrize(
    ("chosen", "rejected", "message"),
    [
        (torch.ones((1, 1)), torch.ones((1, 1)), r"shape \[batch\]"),
        (torch.ones(2), torch.ones(1), "identical shapes"),
        (torch.empty(0), torch.empty(0), "at least one pair"),
        (torch.ones(1, dtype=torch.long), torch.ones(1), "floating point"),
        (torch.tensor([float("nan")]), torch.ones(1), "finite"),
    ],
)
def test_preference_margin_rejects_invalid_inputs(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        preference_margin(chosen, rejected)


def test_preference_margin_rejects_device_mismatch() -> None:
    with pytest.raises(ValueError, match="same device"):
        preference_margin(torch.ones(1), torch.ones(1, device="meta"))


def test_dpo_rejects_invalid_beta_and_batch_alignment() -> None:
    one = torch.ones(1)
    with pytest.raises(ValueError, match="beta must be positive"):
        dpo_loss(one, one, one, one, beta=0)
    with pytest.raises(ValueError, match="policy and reference batches"):
        dpo_loss(torch.ones(2), torch.zeros(2), one, torch.zeros(1), beta=0.1)
