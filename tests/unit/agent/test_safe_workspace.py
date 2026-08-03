from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanopt.agent.tasks import copy_snapshot, load_task_suite
from nanopt.agent.workspace import SafeWorkspace, WorkspacePolicyError, workspace_sha256


@pytest.fixture
def workspace(project_root: Path, tmp_path: Path) -> tuple[SafeWorkspace, Path, Path]:
    task = load_task_suite(project_root / "tasks/mini_swe_v1", split="smoke")[0]
    root = tmp_path / "workspace"
    copy_snapshot(task, root)
    return SafeWorkspace(root, task.card), root, task.oracle_patch_path


def test_bounded_list_read_and_literal_search(workspace: tuple[SafeWorkspace, Path, Path]) -> None:
    safe, _root, _oracle = workspace
    listed = safe.list_files(".", 4)
    assert any(item["path"] == "src/range_utils.py" for item in listed.data["entries"])
    read = safe.read_file("src/range_utils.py", 1, 3)
    assert read.data["lines"][0]["line"] == 1
    found = safe.search("def clamp", "src", "*.py")
    assert found.data["matches"][0]["path"] == "src/range_utils.py"


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "src\\..\\secret", "src/\x00x"])
def test_path_traversal_and_ambiguous_paths_are_rejected(
    workspace: tuple[SafeWorkspace, Path, Path], path: str
) -> None:
    safe, _root, _oracle = workspace
    with pytest.raises(WorkspacePolicyError):
        safe.read_file(path, 1, 2)


def test_symlink_escape_is_rejected(
    workspace: tuple[SafeWorkspace, Path, Path], tmp_path: Path
) -> None:
    safe, root, _oracle = workspace
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    os.symlink(outside, root / "src/link.py")
    with pytest.raises(WorkspacePolicyError, match="symbolic links"):
        safe.read_file("src/link.py", 1, 2)
    with pytest.raises(WorkspacePolicyError, match="symbolic link"):
        workspace_sha256(root)


def test_oracle_patch_is_atomic_and_public_tests_are_protected(
    workspace: tuple[SafeWorkspace, Path, Path],
) -> None:
    safe, root, oracle = workspace
    before = workspace_sha256(root)
    bad_patch = (
        oracle.read_text()
        + "--- a/tests/test_range_utils.py\n"
        + "+++ b/tests/test_range_utils.py\n"
        + "@@ -1,1 +1,1 @@\n"
        + "-import unittest\n"
        + "+raise SystemExit\n"
    )
    with pytest.raises(WorkspacePolicyError, match="protected path"):
        safe.apply_patch(bad_patch)
    assert workspace_sha256(root) == before

    result = safe.apply_patch(oracle.read_text())
    assert result.code == "patched"
    assert workspace_sha256(root) != before


def test_patch_context_mismatch_makes_no_change(
    workspace: tuple[SafeWorkspace, Path, Path],
) -> None:
    safe, root, _oracle = workspace
    before = workspace_sha256(root)
    patch = """--- a/src/range_utils.py
+++ b/src/range_utils.py
@@ -1,1 +1,1 @@
-not the source
+replacement
"""
    with pytest.raises(WorkspacePolicyError, match="context"):
        safe.apply_patch(patch)
    assert workspace_sha256(root) == before
