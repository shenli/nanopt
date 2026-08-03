from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from nanopt.agent.sandbox import FakeSandboxBackend, SandboxLimits
from nanopt.agent.tasks import copy_snapshot, load_task_suite
from nanopt.agent.verifier import HiddenVerifier


def test_public_test_side_effects_do_not_change_submission(
    project_root: Path, tmp_path: Path
) -> None:
    task = load_task_suite(project_root / "tasks/mini_swe_v1", split="smoke")[0]
    workspace = tmp_path / "submission"
    copy_snapshot(task, workspace)
    source = workspace / "src/range_utils.py"
    before = source.read_bytes()
    mutating_card = task.card.model_copy(
        update={
            "public_test_command": [
                "python",
                "-c",
                "from pathlib import Path; Path('src/range_utils.py').write_text('changed')",
            ]
        }
    )

    summary = HiddenVerifier(FakeSandboxBackend(), SandboxLimits(10, 256, 32)).run_public(
        replace(task, card=mutating_card), workspace
    )

    assert summary.status == "passed"
    assert source.read_bytes() == before
