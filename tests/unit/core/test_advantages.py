"""Group-relative advantage tests with small population statistics."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from nanopt.core.advantages import group_relative_advantages


def test_group_zscore_uses_population_standard_deviation() -> None:
    result = group_relative_advantages(torch.tensor([[1.0, 2.0, 3.0]]), epsilon=1e-8)

    population_std = math.sqrt(2.0 / 3.0)
    expected = torch.tensor([[-1.0 / population_std, 0.0, 1.0 / population_std]])
    assert torch.allclose(result.advantages, expected)
    assert result.group_mean.item() == 2.0
    assert result.group_std.item() == pytest.approx(population_std)
    assert result.advantages.sum().item() == pytest.approx(0.0, abs=1e-7)
    assert not result.degenerate_groups.item()


def test_group_centered_does_not_scale_by_standard_deviation() -> None:
    result = group_relative_advantages(
        torch.tensor([[1.0, 2.0, 3.0]]),
        mode="group_centered",
    )

    assert torch.equal(result.advantages, torch.tensor([[-1.0, 0.0, 1.0]]))


def test_equal_rewards_produce_exact_zero_advantages() -> None:
    result = group_relative_advantages(torch.tensor([[4.0, 4.0], [1.0, 3.0]]))

    assert torch.equal(result.advantages[0], torch.zeros(2))
    assert torch.equal(result.degenerate_groups, torch.tensor([True, False]))


@pytest.mark.parametrize(
    ("rewards", "kwargs", "message"),
    [
        (torch.ones(3), {}, r"shape \[batch, group\]"),
        (torch.empty((0, 2)), {}, "must not be empty"),
        (torch.ones((1, 1)), {}, "at least 2"),
        (torch.ones((1, 2), dtype=torch.long), {}, "floating-point dtype"),
        (torch.tensor([[0.0, float("inf")]]), {}, "finite"),
        (torch.ones((1, 2)), {"mode": "unknown"}, "unknown advantage mode"),
        (torch.ones((1, 2)), {"epsilon": 0}, "epsilon must be positive"),
    ],
)
def test_group_advantages_reject_invalid_contracts(
    rewards: torch.Tensor,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        group_relative_advantages(rewards, **kwargs)


def test_group_advantages_return_fp32_for_bf16_rewards() -> None:
    result = group_relative_advantages(torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16))

    assert result.advantages.dtype == torch.float32
