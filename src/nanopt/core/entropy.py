"""Full-vocabulary categorical entropy in explicit FP32 arithmetic."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def categorical_entropy(logits: Tensor) -> Tensor:
    """Return Shannon entropy for every categorical logit vector.

    ``logits`` must be floating point with shape ``[..., vocabulary]``. The result is FP32 with
    shape ``[...]`` and uses natural logarithms, so its unit is nats. Both softmax and the final
    vocabulary reduction run in FP32, even when the input is BF16 or FP16.
    """

    if logits.ndim < 1:
        raise ValueError("logits must have at least a vocabulary dimension")
    if logits.shape[-1] == 0:
        raise ValueError("vocabulary dimension must not be empty")
    if not logits.is_floating_point():
        raise TypeError(f"logits must have a floating-point dtype, got {logits.dtype}")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must contain only finite values")

    log_probs = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    return -(probabilities * log_probs).sum(dim=-1)
