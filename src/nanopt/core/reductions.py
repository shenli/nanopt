"""Explicit FP32 reductions whose denominators come from binary masks."""

from __future__ import annotations

import torch
from torch import Tensor

ReductionDim = int | tuple[int, ...]


def _validate_inputs(values: Tensor, mask: Tensor) -> None:
    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask must have identical shapes, got {tuple(values.shape)} and "
            f"{tuple(mask.shape)}"
        )
    if values.ndim == 0:
        raise ValueError("masked reductions require at least one tensor dimension")
    if mask.is_complex():
        raise TypeError("mask must have a boolean or real numeric dtype")
    if mask.is_floating_point() and not bool(torch.isfinite(mask).all().item()):
        raise ValueError("mask must contain only finite binary values")
    if not bool(torch.logical_or(mask == 0, mask == 1).all().item()):
        raise ValueError("mask must contain only 0 and 1")


def _validate_dims(dim: ReductionDim, ndim: int) -> None:
    dims = (dim,) if isinstance(dim, int) else dim
    if not dims:
        raise ValueError("dim tuple must not be empty")
    normalized = tuple(item + ndim if item < 0 else item for item in dims)
    if any(item < 0 or item >= ndim for item in normalized):
        raise ValueError(f"reduction dim {dim!r} is invalid for a {ndim}-D tensor")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"reduction dim {dim!r} contains a duplicate dimension")


def masked_sum(values: Tensor, mask: Tensor, dim: ReductionDim) -> Tensor:
    """Sum active values in FP32 along one or more dimensions.

    ``values`` and ``mask`` must have identical shapes. The mask may be boolean or numeric 0/1.
    The returned tensor is always FP32, even when ``values`` contains BF16 or FP16 model output.
    Masked positions are multiplied by zero and do not contribute to the result.
    """

    _validate_inputs(values, mask)
    _validate_dims(dim, values.ndim)
    return (values.float() * mask.to(dtype=torch.float32)).sum(dim=dim)


def masked_mean(values: Tensor, mask: Tensor, dim: ReductionDim) -> Tensor:
    """Average active values in FP32 and reject every zero-token reduction.

    ``values`` and ``mask`` must have identical shapes. If any output element would divide by a
    zero active count, this function raises ``ValueError`` instead of hiding malformed data behind
    an epsilon or NaN. For example, reducing ``[batch, sequence]`` over ``sequence`` requires every
    batch item to contain at least one active token.
    """

    _validate_inputs(values, mask)
    _validate_dims(dim, values.ndim)
    mask_fp32 = mask.to(dtype=torch.float32)
    active_counts = mask_fp32.sum(dim=dim)
    if bool((active_counts == 0).any().item()):
        raise ValueError("masked_mean cannot reduce a slice with zero active values")
    numerator = (values.float() * mask_fp32).sum(dim=dim)
    return numerator / active_counts
