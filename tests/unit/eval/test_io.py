from __future__ import annotations

from pathlib import Path

import pytest

from nanopt.data.arithmetic import generate_task
from nanopt.eval.io import read_arithmetic_tasks
from nanopt.runtime.artifacts import append_jsonl


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
