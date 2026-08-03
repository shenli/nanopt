"""Preference rendering and right-padding with explicit completion masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from nanopt.data.preferences import PreferencePair, RejectionType
from nanopt.models.renderer import ChatRenderer, RenderedSupervisedExample
from nanopt.sft.data import CompletionOnlyCollator, SftBatch


@dataclass(frozen=True)
class RenderedPreferencePair:
    """Chosen and rejected renderings sharing one tokenizer-proven prompt boundary."""

    pair: PreferencePair
    chosen: RenderedSupervisedExample
    rejected: RenderedSupervisedExample


@dataclass(frozen=True)
class CachedReferenceValues:
    """Frozen-reference sequence scores attached to a pair by its immutable ID."""

    chosen_logp: float
    rejected_logp: float
    chosen_active_tokens: int
    rejected_active_tokens: int


@dataclass(frozen=True)
class DpoBatch:
    """One padded preference batch and its frozen-reference sequence scores."""

    pair_ids: tuple[str, ...]
    rejection_types: tuple[RejectionType, ...]
    chosen: SftBatch
    rejected: SftBatch
    reference_chosen_logps: Tensor
    reference_rejected_logps: Tensor

    def to(self, device: torch.device | str) -> DpoBatch:
        return DpoBatch(
            pair_ids=self.pair_ids,
            rejection_types=self.rejection_types,
            chosen=self.chosen.to(device),
            rejected=self.rejected.to(device),
            reference_chosen_logps=self.reference_chosen_logps.to(device),
            reference_rejected_logps=self.reference_rejected_logps.to(device),
        )


def render_preference_pairs(
    pairs: Sequence[PreferencePair], renderer: ChatRenderer
) -> list[RenderedPreferencePair]:
    """Render both candidates independently while preserving exact completion boundaries."""

    if not pairs:
        raise ValueError("at least one preference pair is required")
    rendered: list[RenderedPreferencePair] = []
    for pair in pairs:
        messages = [{"role": "user", "content": pair.prompt}]
        chosen = renderer.render_supervised(messages, pair.chosen)
        rejected = renderer.render_supervised(messages, pair.rejected)
        if chosen.input_ids[: chosen.prompt_length] != rejected.input_ids[: rejected.prompt_length]:
            raise ValueError(f"chosen/rejected prompt tokens differ for pair {pair.pair_id}")
        if chosen.prompt_length != rejected.prompt_length:
            raise ValueError(f"chosen/rejected prompt lengths differ for pair {pair.pair_id}")
        rendered.append(RenderedPreferencePair(pair=pair, chosen=chosen, rejected=rejected))
    return rendered


class PreferenceCollator:
    """Collate DPO pairs without truncating either candidate or its completion mask."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        max_prompt_length: int,
        max_completion_length: int,
        reference_values: Mapping[str, CachedReferenceValues] | None = None,
    ) -> None:
        if max_prompt_length <= 0 or max_completion_length <= 0:
            raise ValueError("DPO prompt and completion length limits must be positive")
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.sequence_collator = CompletionOnlyCollator(
            pad_token_id=pad_token_id,
            max_sequence_length=max_prompt_length + max_completion_length,
        )
        self.reference_values = reference_values

    def _validate_lengths(self, example: RenderedSupervisedExample, pair_id: str) -> None:
        completion_tokens = int(sum(example.action_mask))
        if example.prompt_length > self.max_prompt_length:
            raise ValueError(
                f"pair {pair_id} prompt has {example.prompt_length} tokens, exceeding "
                f"{self.max_prompt_length}; DPO does not silently truncate"
            )
        if completion_tokens > self.max_completion_length:
            raise ValueError(
                f"pair {pair_id} completion has {completion_tokens} active tokens, exceeding "
                f"{self.max_completion_length}; DPO does not silently truncate"
            )

    def __call__(self, examples: Sequence[RenderedPreferencePair]) -> DpoBatch:
        if not examples:
            raise ValueError("cannot collate an empty preference batch")
        chosen_examples: list[RenderedSupervisedExample] = []
        rejected_examples: list[RenderedSupervisedExample] = []
        reference_chosen: list[float] = []
        reference_rejected: list[float] = []
        for example in examples:
            self._validate_lengths(example.chosen, example.pair.pair_id)
            self._validate_lengths(example.rejected, example.pair.pair_id)
            chosen_examples.append(example.chosen)
            rejected_examples.append(example.rejected)
            if self.reference_values is not None:
                try:
                    values = self.reference_values[example.pair.pair_id]
                except KeyError as exc:
                    raise ValueError(
                        f"reference cache is missing pair {example.pair.pair_id}"
                    ) from exc
                chosen_count = int(sum(example.chosen.action_mask[1:]))
                rejected_count = int(sum(example.rejected.action_mask[1:]))
                if (
                    chosen_count != values.chosen_active_tokens
                    or rejected_count != values.rejected_active_tokens
                ):
                    raise ValueError(
                        f"reference cache token counts do not match pair {example.pair.pair_id}"
                    )
                reference_chosen.append(values.chosen_logp)
                reference_rejected.append(values.rejected_logp)
        if self.reference_values is None:
            reference_chosen = [0.0] * len(examples)
            reference_rejected = [0.0] * len(examples)
        return DpoBatch(
            pair_ids=tuple(example.pair.pair_id for example in examples),
            rejection_types=tuple(example.pair.rejection_type for example in examples),
            chosen=self.sequence_collator(chosen_examples),
            rejected=self.sequence_collator(rejected_examples),
            reference_chosen_logps=torch.tensor(reference_chosen, dtype=torch.float32),
            reference_rejected_logps=torch.tensor(reference_rejected, dtype=torch.float32),
        )
