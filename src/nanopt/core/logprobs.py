"""Causal token and completion sequence log-probability primitives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from nanopt.core.masks import causal_action_mask
from nanopt.core.reductions import masked_sum

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def causal_token_logps(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Select the log probability of every causally predicted input token.

    Args:
        logits: Floating-point tensor with shape ``[batch, sequence, vocabulary]``. Logits at
            position ``t`` predict the token at ``t + 1``.
        input_ids: Integer tensor with shape ``[batch, sequence]`` on the same device as ``logits``.

    Returns:
        FP32 tensor with shape ``[batch, sequence - 1]``. Result ``[i, j]`` is the log probability
        assigned to ``input_ids[i, j + 1]`` by ``logits[i, j]``. The first input token has no
        returned value because no preceding position predicts it.

    ``log_softmax`` runs in FP32 even when model logits are BF16 or FP16. This function does not
    apply a mask; keeping token selection separate from reduction makes prompt and padding behavior
    inspectable by every caller.
    """

    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape [batch, sequence, vocabulary], got {tuple(logits.shape)}"
        )
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
        )
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            "logits batch/sequence dimensions must match input_ids, "
            f"got {tuple(logits.shape[:2])} and {tuple(input_ids.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("sequence length must be at least 2 for causal token probabilities")
    if logits.shape[2] == 0:
        raise ValueError("vocabulary dimension must not be empty")
    if not logits.is_floating_point():
        raise TypeError(f"logits must have a floating-point dtype, got {logits.dtype}")
    if input_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"input_ids must have an integer dtype, got {input_ids.dtype}")
    if input_ids.device != logits.device:
        raise ValueError("input_ids and logits must be on the same device")
    vocabulary_size = logits.shape[2]
    if bool(torch.logical_or(input_ids < 0, input_ids >= vocabulary_size).any().item()):
        raise ValueError(f"input_ids values must be between 0 and {vocabulary_size - 1}")

    # Position t predicts token t + 1, so the last logits row and first token ID have no partner.
    prediction_logits = logits[:, :-1, :].float()
    target_ids = input_ids[:, 1:].to(torch.int64)
    log_probs = F.log_softmax(prediction_logits, dim=-1)
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


def completion_sequence_logps(logits: Tensor, input_ids: Tensor, action_mask: Tensor) -> Tensor:
    """Sum active completion-token log probabilities for each sequence.

    ``logits`` has shape ``[batch, sequence, vocabulary]``; ``input_ids`` and ``action_mask`` have
    shape ``[batch, sequence]``. The action mask is aligned with full token IDs: one means that the
    token at that exact position belongs to the optimized completion. The function shifts the mask
    once to match causal-token coordinates, then returns an FP32 tensor of shape ``[batch]``.

    A sequence with no active predicted completion tokens raises ``ValueError``. Sequence log
    probability uses a sum, which is the NanoPT reference convention for DPO.
    """

    if action_mask.shape != input_ids.shape:
        raise ValueError(
            "action_mask must match input_ids shape, "
            f"got {tuple(action_mask.shape)} and {tuple(input_ids.shape)}"
        )
    if action_mask.device != logits.device:
        raise ValueError("action_mask and logits must be on the same device")
    token_logps = causal_token_logps(logits, input_ids)
    prediction_mask = causal_action_mask(action_mask)
    if bool((prediction_mask.sum(dim=1) == 0).any().item()):
        raise ValueError("every sequence must contain at least one predicted completion token")
    return masked_sum(token_logps, prediction_mask, dim=1)
