"""Strict JSONL loading for versioned evaluation task records."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from nanopt.data.schemas import ArithmeticTask
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
