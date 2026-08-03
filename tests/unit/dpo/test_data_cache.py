from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanopt.data.preferences import PreferencePair
from nanopt.dpo.cache import (
    ReferenceCacheIdentity,
    build_reference_cache,
    load_reference_cache,
    reference_cache_parity_error,
)
from nanopt.dpo.data import PreferenceCollator, RenderedPreferencePair
from nanopt.models.renderer import RenderedSupervisedExample


class TinyReference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("transition", torch.arange(64, dtype=torch.float32).reshape(8, 8) / 20)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.transition[input_ids])


def _rendered(pair_id: str = "pair-1") -> RenderedPreferencePair:
    pair = PreferencePair(
        pair_id=pair_id,
        task_id="task-1",
        family="addition_subtraction",
        difficulty=1,
        split="train",
        prompt="compute",
        chosen="good",
        rejected="bad",
        rejection_type="wrong_answer",
        source_dataset_fingerprint="source",
        seed=1,
    )

    def sequence(last: int) -> RenderedSupervisedExample:
        return RenderedSupervisedExample(
            input_ids=(1, 2, 3, last),
            attention_mask=(True,) * 4,
            action_mask=(False, False, True, True),
            prompt_length=2,
            chat_template_sha256="a" * 64,
        )

    return RenderedPreferencePair(pair=pair, chosen=sequence(4), rejected=sequence(5))


def _identity() -> ReferenceCacheIdentity:
    return ReferenceCacheIdentity(
        model_id="tiny",
        model_revision="revision",
        tokenizer_revision="tokenizer",
        sft_adapter_sha256="a" * 64,
        chat_template_sha256="b" * 64,
        preference_dataset_fingerprint="c" * 64,
        max_prompt_length=8,
        max_completion_length=8,
    )


def test_reference_cache_round_trip_live_parity_and_invalidation(tmp_path: Path) -> None:
    model = TinyReference()
    examples = [_rendered()]
    collator = PreferenceCollator(pad_token_id=0, max_prompt_length=8, max_completion_length=8)
    path = tmp_path / "cache"

    manifest, values = build_reference_cache(
        model,
        examples,
        collator,
        identity=_identity(),
        output_dir=path,
        micro_batch_size=1,
        device=torch.device("cpu"),
    )
    loaded_manifest, loaded_values = load_reference_cache(path, expected_identity=_identity())

    assert loaded_manifest == manifest
    assert loaded_values == values
    assert (
        reference_cache_parity_error(
            model,
            examples,
            collator,
            values,
            sample_size=1,
            micro_batch_size=1,
            device=torch.device("cpu"),
        )
        == 0
    )
    changed = _identity().model_copy(update={"max_completion_length": 7})
    with pytest.raises(ValueError, match="identity"):
        load_reference_cache(path, expected_identity=changed)


def test_preference_collator_rejects_length_overflow_and_missing_cache() -> None:
    example = _rendered()
    with pytest.raises(ValueError, match="does not silently truncate"):
        PreferenceCollator(pad_token_id=0, max_prompt_length=1, max_completion_length=8)([example])

    with pytest.raises(ValueError, match="missing pair"):
        PreferenceCollator(
            pad_token_id=0,
            max_prompt_length=8,
            max_completion_length=8,
            reference_values={},
        )([example])
