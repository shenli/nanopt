from __future__ import annotations

from pathlib import Path

import pytest

from nanopt.data.arithmetic import ArithmeticGeneratorConfig, generate_task, generate_tasks
from nanopt.data.schemas import ArithmeticTask, SplitManifest, SplitName
from nanopt.data.splits import build_splits
from nanopt.eval.io import (
    read_arithmetic_tasks,
    read_split_manifest,
    validate_tasks_against_manifest,
)
from nanopt.runtime.artifacts import append_jsonl, write_json


def test_read_arithmetic_tasks_round_trips_typed_records(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    task = generate_task(family="multiplication", difficulty=1, seed=8).model_copy(
        update={"split": "test_iid"}
    )
    append_jsonl(path, task.model_dump(mode="json", exclude_none=True))
    assert read_arithmetic_tasks(path) == [task]


def test_read_arithmetic_tasks_rejects_empty_invalid_and_duplicate_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        read_arithmetic_tasks(empty)

    invalid = tmp_path / "invalid.jsonl"
    append_jsonl(invalid, {"task_id": "missing-everything"})
    with pytest.raises(ValueError, match=r"invalid.jsonl:1"):
        read_arithmetic_tasks(invalid)

    duplicate = tmp_path / "duplicate.jsonl"
    task = generate_task(family="multiplication", difficulty=1, seed=9)
    append_jsonl(duplicate, task.model_dump(mode="json", exclude_none=True))
    append_jsonl(duplicate, task.model_dump(mode="json", exclude_none=True))
    with pytest.raises(ValueError, match="duplicate"):
        read_arithmetic_tasks(duplicate)


def _manifest_fixture() -> tuple[list[ArithmeticTask], SplitManifest]:
    config = ArithmeticGeneratorConfig(seed=3, count=7)
    counts: dict[SplitName, int] = {
        "train": 1,
        "validation": 1,
        "test_iid": 1,
        "test_compositional": 1,
        "test_range": 1,
        "test_format_attack": 1,
        "smoke": 1,
    }
    splits, manifest = build_splits(
        generate_tasks(config), counts=counts, seed=4, generator_config=config
    )
    tasks = [task for values in splits.values() for task in values]
    return tasks, manifest


def test_split_manifest_round_trip_proves_counts_and_hashes(tmp_path: Path) -> None:
    tasks, manifest = _manifest_fixture()
    path = tmp_path / "dataset_manifest.json"
    write_json(path, manifest.model_dump(mode="json"))
    loaded = read_split_manifest(path)
    validate_tasks_against_manifest(tasks, loaded)

    changed = list(tasks)
    changed[0] = changed[0].model_copy(update={"prompt": "changed"})
    validate_tasks_against_manifest(changed, loaded)  # Prompt text does not define identity.
    changed[0] = changed[0].model_copy(update={"split": "smoke"})
    with pytest.raises(ValueError, match="count for"):
        validate_tasks_against_manifest(changed, loaded)


def test_split_manifest_loader_rejects_missing_or_invalid_document(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid dataset split manifest"):
        read_split_manifest(tmp_path / "missing.json")
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="invalid dataset split manifest"):
        read_split_manifest(path)
