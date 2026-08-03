"""EOS locations and active-token masks in generated-token coordinates."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_generated_ids(generated_ids: Tensor) -> None:
    if generated_ids.ndim != 2:
        raise ValueError(
            "generated_ids must have shape [batch, generated_sequence], "
            f"got {tuple(generated_ids.shape)}"
        )
    if generated_ids.shape[0] == 0 or generated_ids.shape[1] == 0:
        raise ValueError("generated_ids must have non-empty batch and sequence dimensions")
    if generated_ids.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(f"generated_ids must have an integer dtype, got {generated_ids.dtype}")


def first_eos_mask(generated_ids: Tensor, eos_token_id: int) -> Tensor:
    """Mark only the first EOS token in every generated sequence.

    Args:
        generated_ids: Integer tensor shaped ``[batch, generated_sequence]``. It may contain
            padding or arbitrary values after EOS; this function never interprets those values.
        eos_token_id: Token ID that terminates generation.

    Returns:
        Boolean tensor with the same shape and device. A row without EOS is entirely false.

    Keeping the first-EOS location separate from the active mask makes termination behavior easy
    to inspect in tests and saved rollouts.
    """

    _validate_generated_ids(generated_ids)
    if eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative")
    is_eos = generated_ids == eos_token_id
    eos_count = is_eos.to(torch.int64).cumsum(dim=1)
    return is_eos & (eos_count == 1)


def active_through_eos_mask(
    generated_ids: Tensor,
    eos_token_id: int,
    *,
    include_eos: bool = True,
) -> Tensor:
    """Mark generated actions through the first EOS and mask every later position.

    ``generated_ids`` and the returned mask both use generated-token coordinates
    ``[batch, generated_sequence]``. With ``include_eos=True`` the first EOS is an active action,
    matching NanoPT's policy-gradient convention. Rows without EOS remain fully active.
    """

    first = first_eos_mask(generated_ids, eos_token_id)
    terminals_seen = first.to(torch.int64).cumsum(dim=1)
    before = terminals_seen == 0
    return before | (first if include_eos else torch.zeros_like(first))
