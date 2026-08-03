"""Exact and sampled KL fixtures with independently calculated expectations."""

from __future__ import annotations

import math

import pytest
import torch

from nanopt.core.kl import categorical_kl, sampled_direct_kl, sampled_k3_kl


def test_exact_categorical_kl_matches_two_outcome_calculation() -> None:
    policy = torch.tensor([[0.75, 0.25]]).log()
    reference = torch.tensor([[0.5, 0.5]]).log()

    result = categorical_kl(policy, reference)

    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert result.item() == pytest.approx(expected)


def test_exact_kl_is_zero_for_identical_distributions() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5]])

    assert categorical_kl(logits, logits).item() == pytest.approx(0.0, abs=1e-7)


def test_sampled_direct_and_k3_estimators_match_formulas() -> None:
    policy_logps = torch.tensor([math.log(0.4)])
    reference_logps = torch.tensor([math.log(0.2)])

    direct = sampled_direct_kl(policy_logps, reference_logps)
    k3 = sampled_k3_kl(policy_logps, reference_logps)

    expected_direct = math.log(2.0)
    expected_k3 = math.exp(-expected_direct) + expected_direct - 1.0
    assert direct.item() == pytest.approx(expected_direct)
    assert k3.item() == pytest.approx(expected_k3)
    assert k3.item() >= 0


@pytest.mark.parametrize(
    ("policy", "reference", "message"),
    [
        (torch.ones(2), torch.ones(3), "identical shapes"),
        (torch.ones(2, dtype=torch.long), torch.ones(2), "floating point"),
        (torch.tensor([0.0, float("inf")]), torch.zeros(2), "finite"),
    ],
)
def test_sampled_kl_rejects_invalid_pairs(
    policy: torch.Tensor,
    reference: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        sampled_direct_kl(policy, reference)


def test_sampled_kl_rejects_device_mismatch() -> None:
    with pytest.raises(ValueError, match="same device"):
        sampled_direct_kl(torch.ones(2), torch.ones(2, device="meta"))


def test_exact_kl_rejects_missing_vocabulary_dimension() -> None:
    with pytest.raises(ValueError, match="at least a vocabulary"):
        categorical_kl(torch.tensor(1.0), torch.tensor(1.0))
    with pytest.raises(ValueError, match="must not be empty"):
        categorical_kl(torch.empty((1, 0)), torch.empty((1, 0)))


def test_k3_rejects_invalid_or_unsafe_exponent_bound() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        sampled_k3_kl(torch.zeros(1), torch.zeros(1), max_exp_argument=0)
    with pytest.raises(ValueError, match="exceeds diagnostic bound"):
        sampled_k3_kl(torch.tensor([-100.0]), torch.tensor([0.0]))
