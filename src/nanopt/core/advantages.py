"""Group-relative response advantages for GRPO/RLVR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

AdvantageMode = Literal["group_centered", "group_zscore"]


@dataclass(frozen=True)
class GroupAdvantageResult:
    """Advantages and diagnostics for a ``[batch, group]`` reward tensor."""

    advantages: Tensor
    group_mean: Tensor
    group_std: Tensor
    degenerate_groups: Tensor


def group_relative_advantages(
    rewards: Tensor,
    *,
    mode: AdvantageMode = "group_zscore",
    epsilon: float = 1e-8,
) -> GroupAdvantageResult:
    """Center or population-standardize rewards within each prompt group.

    ``rewards`` must have shape ``[batch, group]`` with ``group >= 2``. Results are FP32. In
    ``group_centered`` mode, each reward is reduced by its group mean. In ``group_zscore`` mode, the
    centered reward is divided by population standard deviation plus ``epsilon``. Equal-reward
    groups produce exactly zero advantages and are marked in ``degenerate_groups``.
    """

    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [batch, group], got {tuple(rewards.shape)}")
    if rewards.shape[0] == 0:
        raise ValueError("rewards batch must not be empty")
    if rewards.shape[1] < 2:
        raise ValueError("group size must be at least 2")
    if not rewards.is_floating_point():
        raise TypeError(f"rewards must have a floating-point dtype, got {rewards.dtype}")
    if not bool(torch.isfinite(rewards).all().item()):
        raise ValueError("rewards must contain only finite values")
    if mode not in {"group_centered", "group_zscore"}:
        raise ValueError(f"unknown advantage mode: {mode!r}")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    rewards_fp32 = rewards.float()
    group_mean = rewards_fp32.mean(dim=1, keepdim=True)
    centered = rewards_fp32 - group_mean
    # unbiased=False is the population standard deviation required by the reference formula.
    group_std = rewards_fp32.std(dim=1, unbiased=False, keepdim=True)
    degenerate_groups = group_std.squeeze(1) == 0
    advantages = centered if mode == "group_centered" else centered / (group_std + epsilon)
    return GroupAdvantageResult(
        advantages=advantages,
        group_mean=group_mean.squeeze(1),
        group_std=group_std.squeeze(1),
        degenerate_groups=degenerate_groups,
    )
