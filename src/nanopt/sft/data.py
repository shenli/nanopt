"""Render trusted arithmetic solutions and pad them without changing the loss boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from nanopt.data.arithmetic import render_trusted_completion
from nanopt.data.schemas import ArithmeticTask
from nanopt.models.renderer import ChatRenderer, RenderedSupervisedExample


@dataclass(frozen=True)
class SftBatch:
    """One right-padded completion-only batch in full-token coordinates.

    All tensors have shape ``[batch, sequence]``. ``action_mask`` marks completion target tokens;
    prompt and padding positions are false. The causal objective shifts this mask exactly once.
    """

    input_ids: Tensor
    attention_mask: Tensor
    action_mask: Tensor

    def to(self, device: torch.device | str) -> SftBatch:
        """Move all batch tensors together so masks cannot stay on the wrong device."""

        return SftBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
        )


def render_sft_examples(
    tasks: Sequence[ArithmeticTask], renderer: ChatRenderer
) -> list[RenderedSupervisedExample]:
    """Render trusted targets while preserving the tokenizer-proven prompt boundary."""

    if not tasks:
        raise ValueError("at least one SFT task is required")
    return [
        renderer.render_supervised(
            [{"role": "user", "content": task.prompt}],
            render_trusted_completion(task),
        )
        for task in tasks
    ]


class CompletionOnlyCollator:
    """Right-pad rendered examples and keep prompt tokens out of the SFT objective.

    NanoPT rejects overlong examples instead of silently truncating a target or moving the
    prompt/completion boundary. Dynamic padding is safe because both attention and action masks
    receive false values in every added position.
    """

    def __init__(self, *, pad_token_id: int, max_sequence_length: int) -> None:
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length

    def __call__(self, examples: Sequence[RenderedSupervisedExample]) -> SftBatch:
        if not examples:
            raise ValueError("cannot collate an empty SFT batch")
        maximum = max(len(example.input_ids) for example in examples)
        if maximum > self.max_sequence_length:
            raise ValueError(
                f"rendered sequence length {maximum} exceeds configured maximum "
                f"{self.max_sequence_length}; SFT does not silently truncate examples"
            )
        if maximum < 2:
            raise ValueError("SFT sequences must contain at least two tokens")

        input_rows: list[list[int]] = []
        attention_rows: list[list[bool]] = []
        action_rows: list[list[bool]] = []
        for example in examples:
            length = len(example.input_ids)
            if not (
                length == len(example.attention_mask) == len(example.action_mask)
                and 0 < example.prompt_length < length
            ):
                raise ValueError("rendered SFT example has inconsistent token or boundary lengths")
            if not any(example.action_mask[1:]):
                raise ValueError("every SFT example needs a causally predicted completion token")
            padding = maximum - length
            input_rows.append([*example.input_ids, *([self.pad_token_id] * padding)])
            attention_rows.append([*example.attention_mask, *([False] * padding)])
            action_rows.append([*example.action_mask, *([False] * padding)])

        return SftBatch(
            input_ids=torch.tensor(input_rows, dtype=torch.long),
            attention_mask=torch.tensor(attention_rows, dtype=torch.bool),
            action_mask=torch.tensor(action_rows, dtype=torch.bool),
        )
