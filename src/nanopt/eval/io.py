"""Strict JSONL loading for versioned evaluation task records."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from nanopt.data.fingerprints import canonical_task_hash
from nanopt.data.schemas import ArithmeticTask, SplitManifest
from nanopt.runtime.artifacts import read_jsonl


def read_arithmetic_tasks(path: Path) -> list[ArithmeticTask]:
    """Load non-empty typed tasks and identify schema failures by record index."""

    tasks: list[ArithmeticTask] = []
    for index, value in enumerate(read_jsonl(path), start=1):
        try:
            tasks.append(ArithmeticTask.model_validate(value, strict=True))
        except ValidationError as exc:
            raise ValueError(f"invalid arithmetic task at {path}:{index}: {exc}") from exc
    if not tasks:
        raise ValueError(f"task file is empty: {path}")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError(f"task file contains duplicate task IDs: {path}")
    return tasks


def read_split_manifest(path: Path) -> SplitManifest:
    """Load the typed split manifest paired with one generated task JSONL file."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return SplitManifest.model_validate(value, strict=True)
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(f"invalid dataset split manifest {path}: {exc}") from exc


def validate_tasks_against_manifest(tasks: list[ArithmeticTask], manifest: SplitManifest) -> None:
    """Prove task counts and canonical hashes match the recorded split manifest exactly."""

    if sum(manifest.counts.values()) != len(tasks):
        raise ValueError("dataset manifest counts do not match the task record count")
    for split, expected_count in manifest.counts.items():
        split_tasks = [task for task in tasks if task.split == split]
        if len(split_tasks) != expected_count:
            raise ValueError(
                f"dataset manifest count for {split} is {expected_count}, "
                f"but the task file contains {len(split_tasks)}"
            )
        hashes = [canonical_task_hash(task) for task in split_tasks]
        if hashes != manifest.canonical_hashes[split]:
            raise ValueError(f"dataset manifest canonical hashes do not match split {split}")
