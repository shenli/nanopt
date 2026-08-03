"""Inspect one immutable MiniSWE task card and its reset boundary."""

from __future__ import annotations

import tempfile
from pathlib import Path

from nanopt.agent.tasks import copy_snapshot, load_task_suite
from nanopt.agent.workspace import SafeWorkspace, WorkspacePolicyError


def main() -> None:
    """Load, reset, and challenge one original task's path policy."""

    task = load_task_suite(Path("tasks/mini_swe_v1"), split="smoke")[0]
    with tempfile.TemporaryDirectory(prefix="nanopt-task-card-lab-") as temporary:
        workspace_root = Path(temporary) / "workspace"
        copied_hash = copy_snapshot(task, workspace_root)
        workspace = SafeWorkspace(workspace_root, task.card)
        try:
            workspace.read_file("../hidden_tests/test_hidden.py", 1, 20)
        except WorkspacePolicyError as exc:
            violation = exc.code
        else:
            raise AssertionError("path traversal unexpectedly succeeded")

    print(f"Task:             {task.card.id}")
    print(f"Snapshot SHA-256: {copied_hash}")
    print(f"Editable globs:   {task.card.editable_globs}")
    print(f"Protected globs:  {task.card.protected_globs}")
    print(f"Traversal result: {violation}")
    assert copied_hash == task.card.snapshot_sha256
    assert violation == "path_traversal"
    print("Task-card lab passed.")


if __name__ == "__main__":
    main()
