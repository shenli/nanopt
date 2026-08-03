"""Leakage-safe deterministic splitting by canonical task hash."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from nanopt.data.arithmetic import ArithmeticGeneratorConfig
from nanopt.data.fingerprints import canonical_task_hash, dataset_fingerprint
from nanopt.data.schemas import ArithmeticTask, SplitManifest, SplitName

SPLIT_ORDER: tuple[SplitName, ...] = (
    "train",
    "validation",
    "test_iid",
    "test_compositional",
    "test_range",
    "test_format_attack",
    "smoke",
)


def build_splits(
    tasks: Sequence[ArithmeticTask],
    *,
    counts: Mapping[SplitName, int],
    seed: int,
    generator_config: ArithmeticGeneratorConfig,
) -> tuple[dict[SplitName, list[ArithmeticTask]], SplitManifest]:
    """Assign every unique canonical task to exactly one requested split.

    Tasks are ordered by a SHA-256 hash of the split seed and canonical task hash, then sliced using
    ``SPLIT_ORDER``. Prompt text never participates in assignment. The function rejects canonical
    duplicates before splitting, even when their task IDs or rendered prompts differ.
    """

    unknown = set(counts) - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"unknown split names: {', '.join(sorted(unknown))}")
    normalized_counts: dict[SplitName, int] = {
        name: int(counts.get(name, 0)) for name in SPLIT_ORDER
    }
    if any(value < 0 for value in normalized_counts.values()):
        raise ValueError("split counts must be nonnegative")
    if sum(normalized_counts.values()) != len(tasks):
        raise ValueError("split counts must assign every task exactly once")

    hashes = [canonical_task_hash(task) for task in tasks]
    if len(set(hashes)) != len(hashes):
        raise ValueError("canonical task overlap exists before split assignment")
    decorated = sorted(
        zip(tasks, hashes, strict=True),
        key=lambda pair: hashlib.sha256(f"{seed}:{pair[1]}".encode()).hexdigest(),
    )
    splits: dict[SplitName, list[ArithmeticTask]] = {name: [] for name in SPLIT_ORDER}
    canonical_hashes: dict[SplitName, list[str]] = {name: [] for name in SPLIT_ORDER}
    cursor = 0
    for name in SPLIT_ORDER:
        next_cursor = cursor + normalized_counts[name]
        for task, task_hash in decorated[cursor:next_cursor]:
            splits[name].append(task.model_copy(update={"split": name}))
            canonical_hashes[name].append(task_hash)
        cursor = next_cursor

    assigned = [task for name in SPLIT_ORDER for task in splits[name]]
    fingerprint = dataset_fingerprint(assigned, generator_config=generator_config)
    manifest = SplitManifest(
        seed=seed,
        dataset_fingerprint=fingerprint,
        counts=normalized_counts,
        canonical_hashes=canonical_hashes,
    )
    return splits, manifest
