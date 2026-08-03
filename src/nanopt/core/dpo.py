"""Direct Preference Optimization margins and logistic loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class DpoLossResult:
    """Inspectable per-pair values and scalar batch loss for one DPO calculation."""

    loss: Tensor
    per_pair_loss: Tensor
    policy_margin: Tensor
    reference_margin: Tensor
    implicit_reward_margin: Tensor


def preference_margin(chosen_logps: Tensor, rejected_logps: Tensor) -> Tensor:
    """Return ``chosen_logps - rejected_logps`` in FP32.

    Inputs are one sequence log probability per preference pair with shape ``[batch]``. They must
    already use the same completion-mask and sequence-reduction convention.
    """

    if chosen_logps.ndim != 1 or rejected_logps.ndim != 1:
        raise ValueError("chosen and rejected log probabilities must have shape [batch]")
    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError(
            "chosen and rejected log probabilities must have identical shapes, "
            f"got {tuple(chosen_logps.shape)} and {tuple(rejected_logps.shape)}"
        )
    if chosen_logps.numel() == 0:
        raise ValueError("a DPO batch must contain at least one pair")
    if not chosen_logps.is_floating_point() or not rejected_logps.is_floating_point():
        raise TypeError("chosen and rejected log probabilities must be floating point")
    if chosen_logps.device != rejected_logps.device:
        raise ValueError("chosen and rejected log probabilities must be on the same device")
    if not bool(torch.isfinite(chosen_logps).all().item()) or not bool(
        torch.isfinite(rejected_logps).all().item()
    ):
        raise ValueError("chosen and rejected log probabilities must contain only finite values")
    return chosen_logps.float() - rejected_logps.float()


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    *,
    beta: float,
) -> DpoLossResult:
    """Compute the standard mean DPO loss for a batch of preference pairs.

    Every input has shape ``[batch]`` and contains a completion sequence log-probability, normally
    a masked sum over active tokens. The policy and frozen-reference margins are subtracted, scaled
    by positive ``beta``, and passed through ``-logsigmoid``. All arithmetic and returned tensors
    are FP32.
    """

    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_margin = preference_margin(policy_chosen_logps, policy_rejected_logps)
    reference_margin = preference_margin(reference_chosen_logps, reference_rejected_logps)
    if policy_margin.shape != reference_margin.shape:
        raise ValueError(
            "policy and reference batches must have identical shapes, "
            f"got {tuple(policy_margin.shape)} and {tuple(reference_margin.shape)}"
        )
    if policy_margin.device != reference_margin.device:
        raise ValueError("policy and reference batches must be on the same device")

    implicit_reward_margin = beta * (policy_margin - reference_margin)
    per_pair_loss = -F.logsigmoid(implicit_reward_margin)
    return DpoLossResult(
        loss=per_pair_loss.mean(),
        per_pair_loss=per_pair_loss,
        policy_margin=policy_margin,
        reference_margin=reference_margin,
        implicit_reward_margin=implicit_reward_margin,
    )
