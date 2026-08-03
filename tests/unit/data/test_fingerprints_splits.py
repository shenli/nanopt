"""Dataset fingerprints and canonical-hash split leakage tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from nanopt.data.arithmetic import ArithmeticGeneratorConfig, generate_tasks
from nanopt.data.fingerprints import canonical_task_hash, dataset_fingerprint
from nanopt.data.schemas import SplitName
from nanopt.data.splits import SPLIT_ORDER, build_splits


def _counts() -> dict[SplitName, int]:
    return {
        "train": 8,
        "validation": 4,
        "test_iid": 2,
        "test_compositional": 2,
        "test_range": 2,
        "test_format_attack": 1,
        "smoke": 1,
    }


def test_dataset_fingerprint_repeats_for_identical_seed_and_content() -> None:
    config = ArithmeticGeneratorConfig(seed=42, count=20)

    first = dataset_fingerprint(generate_tasks(config), generator_config=config)
    second = dataset_fingerprint(generate_tasks(config), generator_config=config)
    changed_config = ArithmeticGeneratorConfig(seed=43, count=20)
    changed = dataset_fingerprint(generate_tasks(changed_config), generator_config=changed_config)

    assert first == second
    assert len(first) == 64
    assert changed != first


def test_split_builder_is_deterministic_complete_and_disjoint(project_root: Path) -> None:
    config = ArithmeticGeneratorConfig(seed=42, count=20)
    tasks = generate_tasks(config)

    first_splits, first_manifest = build_splits(
        tasks,
        counts=_counts(),
        seed=7,
        generator_config=config,
    )
    second_splits, second_manifest = build_splits(
        tasks,
        counts=_counts(),
        seed=7,
        generator_config=config,
    )

    assert first_manifest == second_manifest
    assert first_manifest.counts == _counts()
    assert [task.task_id for task in first_splits["train"]] == [
        task.task_id for task in second_splits["train"]
    ]
    all_hashes = [value for name in SPLIT_ORDER for value in first_manifest.canonical_hashes[name]]
    assert len(all_hashes) == len(set(all_hashes)) == 20
    for name in SPLIT_ORDER:
        assert all(task.split == name for task in first_splits[name])

    schema = json.loads((project_root / "specs/schemas/dataset_manifest.schema.json").read_text())
    jsonschema.validate(first_manifest.model_dump(mode="json"), schema)


def test_split_builder_rejects_canonical_overlap_even_if_text_changes() -> None:
    config = ArithmeticGeneratorConfig(seed=1, count=4)
    tasks = generate_tasks(config)
    duplicate = tasks[0].model_copy(update={"task_id": "different", "prompt": "paraphrased"})

    with pytest.raises(ValueError, match="canonical task overlap"):
        build_splits(
            [*tasks, duplicate],
            counts={"train": 5},
            seed=1,
            generator_config=config,
        )
    assert canonical_task_hash(tasks[0]) == canonical_task_hash(duplicate)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"unknown": 4}, "unknown split"),
        ({"train": -1, "smoke": 5}, "nonnegative"),
        ({"train": 3}, "every task exactly once"),
    ],
)
def test_split_builder_rejects_invalid_count_contracts(
    counts: dict[Any, int],
    message: str,
) -> None:
    config = ArithmeticGeneratorConfig(seed=1, count=4)
    with pytest.raises(ValueError, match=message):
        build_splits(
            generate_tasks(config),
            counts=counts,
            seed=1,
            generator_config=config,
        )


def test_dataset_fingerprint_rejects_empty_and_duplicate_ids() -> None:
    config = ArithmeticGeneratorConfig(seed=1, count=2)
    tasks = generate_tasks(config)
    with pytest.raises(ValueError, match="empty dataset"):
        dataset_fingerprint([], generator_config=config)
    with pytest.raises(ValueError, match="duplicate task IDs"):
        dataset_fingerprint([tasks[0], tasks[0]], generator_config=config)
