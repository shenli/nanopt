"""PPO/GRPO probability ratios and clipped policy loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from nanopt.core.reductions import masked_mean

LossNormalization = Literal["token_mean", "sequence_mean"]


@dataclass(frozen=True)
class ClippedPolicyLossResult:
    """Scalar loss plus token-level diagnostics for an inspectable policy update."""

    loss: Tensor
    ratios: Tensor
    per_token_loss: Tensor
    clipped: Tensor
    clip_fraction: Tensor
    max_abs_log_ratio: Tensor


def probability_ratio(
    current_logps: Tensor,
    old_logps: Tensor,
    *,
    max_abs_log_ratio: float = 60.0,
) -> Tensor:
    """Return ``exp(current_logps - old_logps)`` in FP32.

    Inputs and output have the same shape. ``max_abs_log_ratio`` is a diagnostic safety bound, not
    a clamp. Exceeding it raises before exponentiation so numerical instability remains visible.
    """

    if current_logps.shape != old_logps.shape:
        raise ValueError(
            "current and old log probabilities must have identical shapes, "
            f"got {tuple(current_logps.shape)} and {tuple(old_logps.shape)}"
        )
    if current_logps.numel() == 0:
        raise ValueError("log-probability tensors must not be empty")
    if not current_logps.is_floating_point() or not old_logps.is_floating_point():
        raise TypeError("current and old log probabilities must be floating point")
    if current_logps.device != old_logps.device:
        raise ValueError("current and old log probabilities must be on the same device")
    if not bool(torch.isfinite(current_logps).all().item()) or not bool(
        torch.isfinite(old_logps).all().item()
    ):
        raise ValueError("current and old log probabilities must contain only finite values")
    if max_abs_log_ratio <= 0:
        raise ValueError("max_abs_log_ratio must be positive")

    log_ratio = current_logps.float() - old_logps.float()
    maximum = log_ratio.abs().max()
    if maximum.item() > max_abs_log_ratio:
        raise ValueError(
            f"absolute log-ratio {maximum.item():.4g} exceeds diagnostic bound "
            f"{max_abs_log_ratio:.4g}"
        )
    return log_ratio.exp()


def _broadcast_advantages(advantages: Tensor, token_shape: torch.Size) -> Tensor:
    if advantages.shape == token_shape:
        return advantages
    if advantages.shape == token_shape[:-1]:
        return advantages.unsqueeze(-1).expand(token_shape)
    raise ValueError(
        "advantages must match token shape or omit only its final token dimension, "
        f"got {tuple(advantages.shape)} for {tuple(token_shape)}"
    )


def clipped_policy_loss(
    current_logps: Tensor,
    old_logps: Tensor,
    advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_epsilon: float,
    normalization: LossNormalization = "token_mean",
    max_abs_log_ratio: float = 60.0,
) -> ClippedPolicyLossResult:
    """Compute the masked PPO-style clipped policy loss.

    Token log probabilities and ``action_mask`` have shape ``[..., sequence]``. Response-level
    ``advantages`` may have shape ``[...]`` and are broadcast across the final token dimension, or
    they may already have token shape. ``token_mean`` weights every active token equally;
    ``sequence_mean`` averages within each sequence and then weights every sequence equally.
    Returned values use FP32 except the boolean ``clipped`` mask.
    """

    if current_logps.shape != action_mask.shape:
        raise ValueError(
            "action_mask must match token log-probability shape, "
            f"got {tuple(action_mask.shape)} and {tuple(current_logps.shape)}"
        )
    if current_logps.ndim < 2:
        raise ValueError("clipped policy loss requires at least batch and token dimensions")
    if action_mask.device != current_logps.device or advantages.device != current_logps.device:
        raise ValueError("log probabilities, advantages, and action_mask must share a device")
    if not advantages.is_floating_point():
        raise TypeError("advantages must be floating point")
    if not bool(torch.isfinite(advantages).all().item()):
        raise ValueError("advantages must contain only finite values")
    if clip_epsilon <= 0 or clip_epsilon >= 1:
        raise ValueError("clip_epsilon must be between 0 and 1")
    if normalization not in {"token_mean", "sequence_mean"}:
        raise ValueError(f"unknown loss normalization: {normalization!r}")

    ratios = probability_ratio(
        current_logps,
        old_logps,
        max_abs_log_ratio=max_abs_log_ratio,
    )
    token_advantages = _broadcast_advantages(advantages.float(), current_logps.shape)
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped_objective = ratios * token_advantages
    clipped_objective = clipped_ratios * token_advantages
    per_token_loss = -torch.minimum(unclipped_objective, clipped_objective)
    clipped = unclipped_objective > clipped_objective

    all_dims = tuple(range(per_token_loss.ndim))
    clip_fraction = masked_mean(clipped.float(), action_mask, dim=all_dims)
    if normalization == "token_mean":
        loss = masked_mean(per_token_loss, action_mask, dim=all_dims)
    else:
        per_sequence = masked_mean(per_token_loss, action_mask, dim=-1)
        loss = per_sequence.mean()
    max_log_ratio = (current_logps.float() - old_logps.float()).abs().max()
    return ClippedPolicyLossResult(
        loss=loss,
        ratios=ratios,
        per_token_loss=per_token_loss,
        clipped=clipped,
        clip_fraction=clip_fraction,
        max_abs_log_ratio=max_log_ratio,
    )
