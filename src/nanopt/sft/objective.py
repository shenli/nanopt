"""Completion-only negative log likelihood and token-accuracy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nanopt.core.logprobs import causal_token_logps
from nanopt.core.masks import causal_action_mask
from nanopt.core.reductions import masked_mean


@dataclass(frozen=True)
class SftObjective:
    """Scalar loss plus detached diagnostics for one forward pass."""

    loss: Tensor
    token_accuracy: Tensor
    active_tokens: int


def completion_only_objective(
    logits: Tensor,
    input_ids: Tensor,
    action_mask: Tensor,
) -> SftObjective:
    """Compute token-mean completion NLL from full-token masks.

    ``logits`` has shape ``[batch, sequence, vocabulary]``; ``input_ids`` and ``action_mask`` have
    shape ``[batch, sequence]``. Prompt targets and padding contribute exactly zero. The first
    completion token remains active because the final prompt position causally predicts it.
    """

    if action_mask.shape != input_ids.shape:
        raise ValueError("action_mask must match input_ids shape")
    token_logps = causal_token_logps(logits, input_ids)
    prediction_mask = causal_action_mask(action_mask)
    active_tokens = int(prediction_mask.sum().item())
    if active_tokens == 0:
        raise ValueError("SFT objective requires at least one active completion token")
    loss = -masked_mean(token_logps, prediction_mask, dim=(0, 1))

    predictions = logits[:, :-1, :].detach().argmax(dim=-1)
    targets = input_ids[:, 1:]
    correct = (predictions == targets).to(torch.float32)
    token_accuracy = masked_mean(correct, prediction_mask, dim=(0, 1))
    return SftObjective(loss=loss, token_accuracy=token_accuracy, active_tokens=active_tokens)
