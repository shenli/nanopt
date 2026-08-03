"""Exact categorical and sampled token-level KL divergence utilities."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_pair(policy: Tensor, reference: Tensor, *, quantity: str) -> None:
    if policy.shape != reference.shape:
        raise ValueError(
            f"policy and reference {quantity} must have identical shapes, got "
            f"{tuple(policy.shape)} and {tuple(reference.shape)}"
        )
    if not policy.is_floating_point() or not reference.is_floating_point():
        raise TypeError(f"policy and reference {quantity} must be floating point")
    if policy.device != reference.device:
        raise ValueError(f"policy and reference {quantity} must be on the same device")
    if not bool(torch.isfinite(policy).all().item()) or not bool(
        torch.isfinite(reference).all().item()
    ):
        raise ValueError(f"policy and reference {quantity} must contain only finite values")


def categorical_kl(policy_logits: Tensor, reference_logits: Tensor) -> Tensor:
    """Compute exact ``KL(policy || reference)`` over the vocabulary.

    Both inputs have shape ``[..., vocabulary]``. The result has shape ``[...]``, is measured in
    nats, and is computed entirely in FP32. This is a full-distribution diagnostic, unlike sampled
    estimators that inspect only an action selected from the policy.
    """

    _validate_pair(policy_logits, reference_logits, quantity="logits")
    if policy_logits.ndim < 1:
        raise ValueError("logits must have at least a vocabulary dimension")
    if policy_logits.shape[-1] == 0:
        raise ValueError("vocabulary dimension must not be empty")

    policy_log_probs = F.log_softmax(policy_logits.float(), dim=-1)
    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    policy_probs = policy_log_probs.exp()
    return (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1)


def sampled_direct_kl(policy_logps: Tensor, reference_logps: Tensor) -> Tensor:
    """Return the sampled log-ratio ``log(policy) - log(reference)`` in FP32.

    Inputs and output have the same shape, commonly ``[batch, actions]``. One sampled value may be
    negative; only its expectation under policy samples equals the nonnegative KL divergence.
    """

    _validate_pair(policy_logps, reference_logps, quantity="log probabilities")
    return policy_logps.float() - reference_logps.float()


def sampled_k3_kl(
    policy_logps: Tensor,
    reference_logps: Tensor,
    *,
    max_exp_argument: float = 60.0,
) -> Tensor:
    """Return the nonnegative sampled k3 estimator in FP32.

    Let ``d = log(policy) - log(reference)``. The estimator is ``exp(-d) + d - 1`` and has the same
    shape as its inputs. ``max_exp_argument`` is a diagnostic bound, not a clamp: the function
    raises before exponentiation when ``-d`` exceeds it, preventing a silent FP32 overflow.
    """

    if max_exp_argument <= 0:
        raise ValueError("max_exp_argument must be positive")
    log_ratio = sampled_direct_kl(policy_logps, reference_logps)
    exponent = -log_ratio
    if bool((exponent > max_exp_argument).any().item()):
        maximum = exponent.max().item()
        raise ValueError(
            f"sampled k3 exponent {maximum:.4g} exceeds diagnostic bound {max_exp_argument:.4g}"
        )
    return exponent.exp() + log_ratio - 1.0
