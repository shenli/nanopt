"""Construct completion masks in token and causal-prediction coordinates."""

from __future__ import annotations

import torch
from torch import Tensor

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_binary_mask(mask: Tensor, *, name: str) -> None:
    """Reject masks whose values cannot be interpreted unambiguously as active/inactive."""

    if (
        mask.dtype != torch.bool
        and mask.dtype not in _INTEGER_DTYPES
        and not mask.is_floating_point()
    ):
        raise TypeError(f"{name} must have a boolean or numeric dtype, got {mask.dtype}")
    if mask.is_floating_point() and not bool(torch.isfinite(mask).all().item()):
        raise ValueError(f"{name} must contain only finite binary values")
    if not bool(torch.logical_or(mask == 0, mask == 1).all().item()):
        raise ValueError(f"{name} must contain only 0 and 1")


def completion_action_mask(
    attention_mask: Tensor,
    completion_starts: Tensor,
    *,
    terminal_mask: Tensor | None = None,
    include_terminal: bool = True,
) -> Tensor:
    """Build a boolean completion mask aligned with full token IDs.

    Args:
        attention_mask: Shape ``[batch, sequence]``. One marks a real token and zero marks padding.
        completion_starts: Shape ``[batch]`` integer indices. Each value is the position of the
            first completion token in full-token coordinates. ``sequence`` is allowed and creates
            an empty completion, which downstream reductions reject clearly.
        terminal_mask: Optional shape ``[batch, sequence]`` binary mask marking EOS or another
            terminal token. Tokens after the first marked terminal are always inactive.
        include_terminal: Whether the first marked terminal itself is an action token.

    Returns:
        A boolean tensor of shape ``[batch, sequence]`` on the same device as ``attention_mask``.

    Prompt positions, padding, and positions after the first terminal token are false. This
    function works in input-token coordinates; call :func:`causal_action_mask` before combining
    the result with causal token log probabilities.
    """

    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [batch, sequence], got {tuple(attention_mask.shape)}"
        )
    batch_size, sequence_length = attention_mask.shape
    if batch_size == 0 or sequence_length == 0:
        raise ValueError("attention_mask must have non-empty batch and sequence dimensions")
    if completion_starts.shape != (batch_size,):
        raise ValueError(
            "completion_starts must have shape [batch], "
            f"got {tuple(completion_starts.shape)} for batch {batch_size}"
        )
    if completion_starts.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"completion_starts must have an integer dtype, got {completion_starts.dtype}"
        )
    if completion_starts.device != attention_mask.device:
        raise ValueError("completion_starts and attention_mask must be on the same device")
    if bool(
        torch.logical_or(completion_starts < 0, completion_starts > sequence_length).any().item()
    ):
        raise ValueError(f"completion_starts values must be between 0 and {sequence_length}")
    _require_binary_mask(attention_mask, name="attention_mask")

    positions = torch.arange(sequence_length, device=attention_mask.device).unsqueeze(0)
    action_mask = positions >= completion_starts.unsqueeze(1)
    action_mask &= attention_mask.bool()

    if terminal_mask is None:
        return action_mask
    if terminal_mask.shape != attention_mask.shape:
        raise ValueError(
            "terminal_mask must match attention_mask shape, "
            f"got {tuple(terminal_mask.shape)} and {tuple(attention_mask.shape)}"
        )
    if terminal_mask.device != attention_mask.device:
        raise ValueError("terminal_mask and attention_mask must be on the same device")
    _require_binary_mask(terminal_mask, name="terminal_mask")
    terminal = terminal_mask.bool()
    if bool(torch.logical_and(terminal, ~attention_mask.bool()).any().item()):
        raise ValueError("terminal_mask cannot mark padded positions")

    # A cumulative count distinguishes the first terminal position from every position after it.
    terminal_count = terminal.to(torch.int64).cumsum(dim=1)
    before_terminal = terminal_count == 0
    first_terminal = terminal & (terminal_count == 1)
    active_through_terminal = before_terminal | (first_terminal & include_terminal)
    return action_mask & active_through_terminal


def causal_action_mask(action_mask: Tensor) -> Tensor:
    """Shift a full-token action mask into causal-prediction coordinates.

    ``action_mask`` has shape ``[batch, sequence]`` and describes ``input_ids`` positions.
    The returned boolean mask has shape ``[batch, sequence - 1]``; element ``[i, j]`` describes
    the token ``input_ids[i, j + 1]`` predicted by logits at position ``j``.
    """

    if action_mask.ndim != 2:
        raise ValueError(
            f"action_mask must have shape [batch, sequence], got {tuple(action_mask.shape)}"
        )
    if action_mask.shape[1] < 2:
        raise ValueError("action_mask sequence length must be at least 2 for causal shifting")
    _require_binary_mask(action_mask, name="action_mask")
    return action_mask[:, 1:].bool()
