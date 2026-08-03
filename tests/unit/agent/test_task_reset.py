from __future__ import annotations

from pathlib import Path

import pytest

from nanopt.agent.tasks import copy_snapshot, load_agent_task, load_task_suite
from nanopt.sft.checkpoint import sha256_directory


def test_every_original_task_has_stable_reset_hash(project_root: Path, tmp_path: Path) -> None:
    tasks = load_task_suite(project_root / "tasks/mini_swe_v1", split="smoke")
    assert len(tasks) == 5
    for index, task in enumerate(tasks):
        first = tmp_path / f"first-{index}"
        second = tmp_path / f"second-{index}"
        assert copy_snapshot(task, first) == task.card.snapshot_sha256
        assert copy_snapshot(task, second) == task.card.snapshot_sha256
        assert sha256_directory(first) == sha256_directory(second)
        assert not (first / ".nanopt_hidden_tests").exists()


def test_task_loader_rejects_tampered_snapshot(project_root: Path, tmp_path: Path) -> None:
    source = project_root / "tasks/mini_swe_v1/clamp_reversed_bounds"
    import shutil

    copied = tmp_path / source.name
    shutil.copytree(source, copied)
    (copied / "snapshot/src/range_utils.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="snapshot hash differs"):
        load_agent_task(copied)
