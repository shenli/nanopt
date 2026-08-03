"""Fingerprint and persist frozen-reference DPO sequence log probabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from nanopt.core.logprobs import completion_sequence_logps
from nanopt.dpo.data import (
    CachedReferenceValues,
    PreferenceCollator,
    RenderedPreferencePair,
)
from nanopt.runtime.artifacts import (
    append_jsonl,
    canonical_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json,
)


class CacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReferenceCacheEntry(CacheRecord):
    schema_version: Literal[1] = 1
    pair_id: str
    chosen_logp: float
    rejected_logp: float
    chosen_active_tokens: int = Field(gt=0)
    rejected_active_tokens: int = Field(gt=0)


class ReferenceCacheIdentity(CacheRecord):
    """Every input whose change invalidates frozen-reference scores."""

    schema_version: Literal[1] = 1
    model_id: str
    model_revision: str
    tokenizer_revision: str
    sft_adapter_sha256: str
    renderer_version: Literal["chat-renderer-v1"] = "chat-renderer-v1"
    chat_template_sha256: str
    preference_dataset_fingerprint: str
    max_prompt_length: int = Field(gt=0)
    max_completion_length: int = Field(gt=0)
    truncation_policy: Literal["reject"] = "reject"
    eos_inclusion_policy: Literal["include-chat-terminator"] = "include-chat-terminator"
    sequence_logprob_reduction: Literal["sum"] = "sum"
    forward_layout: Literal["concatenated", "separate"]

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


class ReferenceCacheManifest(CacheRecord):
    schema_version: Literal[1] = 1
    identity: ReferenceCacheIdentity
    identity_sha256: str
    entries_path: Literal["reference_logps.jsonl"] = "reference_logps.jsonl"
    entries_sha256: str
    entry_count: int = Field(gt=0)
    cache_sha256: str


def score_reference_batch(
    model: Any,
    batch: Any,
    *,
    concatenate_chosen_rejected: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score chosen and rejected completions under one frozen reference policy."""

    if concatenate_chosen_rejected:
        # Import here to keep the cache record types independent from training orchestration while
        # guaranteeing reference and policy scores use the identical BF16 forward layout.
        from nanopt.dpo.trainer import policy_sequence_logps

        return policy_sequence_logps(model, batch, concatenate_chosen_rejected=True)
    chosen_logits = model(
        input_ids=batch.chosen.input_ids,
        attention_mask=batch.chosen.attention_mask,
        use_cache=False,
    ).logits
    rejected_logits = model(
        input_ids=batch.rejected.input_ids,
        attention_mask=batch.rejected.attention_mask,
        use_cache=False,
    ).logits
    return (
        completion_sequence_logps(chosen_logits, batch.chosen.input_ids, batch.chosen.action_mask),
        completion_sequence_logps(
            rejected_logits, batch.rejected.input_ids, batch.rejected.action_mask
        ),
    )


def build_reference_cache(
    model: Any,
    examples: Sequence[RenderedPreferencePair],
    collator: PreferenceCollator,
    *,
    identity: ReferenceCacheIdentity,
    output_dir: Path,
    micro_batch_size: int,
    concatenate_chosen_rejected: bool,
    device: torch.device,
) -> tuple[ReferenceCacheManifest, dict[str, CachedReferenceValues]]:
    """Compute and persist complete FP32 reference scores before DPO optimization."""

    if not examples:
        raise ValueError("reference cache requires at least one preference pair")
    if micro_batch_size <= 0:
        raise ValueError("reference cache micro-batch size must be positive")
    if output_dir.exists():
        raise ValueError(f"reference cache output must be new: {output_dir}")
    output_dir.mkdir(parents=True)
    entries_path = output_dir / "reference_logps.jsonl"
    was_training = bool(model.training)
    model.eval()
    entries: list[ReferenceCacheEntry] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(examples), micro_batch_size):
                selected = examples[start : start + micro_batch_size]
                batch = collator(selected).to(device)
                chosen, rejected = score_reference_batch(
                    model,
                    batch,
                    concatenate_chosen_rejected=concatenate_chosen_rejected,
                )
                for index, example in enumerate(selected):
                    entry = ReferenceCacheEntry(
                        pair_id=example.pair.pair_id,
                        chosen_logp=float(chosen[index].item()),
                        rejected_logp=float(rejected[index].item()),
                        chosen_active_tokens=int(batch.chosen.action_mask[index, 1:].sum().item()),
                        rejected_active_tokens=int(
                            batch.rejected.action_mask[index, 1:].sum().item()
                        ),
                    )
                    append_jsonl(entries_path, entry.model_dump(mode="json"))
                    entries.append(entry)
    finally:
        if was_training:
            model.train()
    entry_ids = [entry.pair_id for entry in entries]
    if len(entries) != len(examples) or len(set(entry_ids)) != len(entry_ids):
        raise RuntimeError("reference cache is incomplete or contains duplicate pair IDs")
    entries_sha = sha256_file(entries_path)
    cache_sha = sha256_bytes(
        canonical_json(
            {
                "identity_sha256": identity.fingerprint,
                "entries_sha256": entries_sha,
                "entry_count": len(entries),
            }
        )
    )
    manifest = ReferenceCacheManifest(
        identity=identity,
        identity_sha256=identity.fingerprint,
        entries_sha256=entries_sha,
        entry_count=len(entries),
        cache_sha256=cache_sha,
    )
    write_json(output_dir / "cache_manifest.json", manifest.model_dump(mode="json"))
    return manifest, {
        entry.pair_id: CachedReferenceValues(
            chosen_logp=entry.chosen_logp,
            rejected_logp=entry.rejected_logp,
            chosen_active_tokens=entry.chosen_active_tokens,
            rejected_active_tokens=entry.rejected_active_tokens,
        )
        for entry in entries
    }


def load_reference_cache(
    path: Path,
    *,
    expected_identity: ReferenceCacheIdentity,
) -> tuple[ReferenceCacheManifest, dict[str, CachedReferenceValues]]:
    """Validate identity and file hashes before returning cached scores."""

    manifest = ReferenceCacheManifest.model_validate_json(
        (path / "cache_manifest.json").read_text(encoding="utf-8"), strict=True
    )
    if (
        manifest.identity != expected_identity
        or manifest.identity_sha256 != expected_identity.fingerprint
    ):
        raise ValueError("reference cache identity does not match current DPO inputs")
    entries_path = path / manifest.entries_path
    if sha256_file(entries_path) != manifest.entries_sha256:
        raise ValueError("reference cache entries hash does not match its manifest")
    records = [
        ReferenceCacheEntry.model_validate(value, strict=True) for value in read_jsonl(entries_path)
    ]
    if len(records) != manifest.entry_count or len({record.pair_id for record in records}) != len(
        records
    ):
        raise ValueError("reference cache entry count or pair IDs are invalid")
    values: dict[str, CachedReferenceValues] = {
        record.pair_id: CachedReferenceValues(
            chosen_logp=record.chosen_logp,
            rejected_logp=record.rejected_logp,
            chosen_active_tokens=record.chosen_active_tokens,
            rejected_active_tokens=record.rejected_active_tokens,
        )
        for record in records
    }
    return manifest, values


def reference_cache_parity_error(
    model: Any,
    examples: Sequence[RenderedPreferencePair],
    collator: PreferenceCollator,
    cached: Mapping[str, CachedReferenceValues],
    *,
    sample_size: int,
    micro_batch_size: int,
    concatenate_chosen_rejected: bool,
    device: torch.device,
) -> float:
    """Return the largest absolute live/cache difference on a deterministic prefix."""

    if sample_size <= 0:
        raise ValueError("cache validation sample size must be positive")
    if micro_batch_size <= 0:
        raise ValueError("cache validation micro-batch size must be positive")
    selected = examples[: min(sample_size, len(examples))]
    was_training = bool(model.training)
    model.eval()
    differences: list[float] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(selected), micro_batch_size):
                batch_examples = selected[start : start + micro_batch_size]
                batch = collator(batch_examples).to(device)
                live_chosen, live_rejected = score_reference_batch(
                    model,
                    batch,
                    concatenate_chosen_rejected=concatenate_chosen_rejected,
                )
                for index, example in enumerate(batch_examples):
                    value = cached[example.pair.pair_id]
                    differences.extend(
                        (
                            abs(float(live_chosen[index].item()) - value.chosen_logp),
                            abs(float(live_rejected[index].item()) - value.rejected_logp),
                        )
                    )
    finally:
        if was_training:
            model.train()
    return max(differences)
