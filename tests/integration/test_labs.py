"""Keep learner-facing CPU lab commands executable as the implementation evolves."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("lab", "success_message"),
    [
        ("labs/00_prerequisites.py", "Prerequisite self-check passed."),
        ("labs/01_tokens_and_masks.py", "NanoPT sequence logp          = -0.9163"),
        ("labs/02_logprob_by_hand.py", "Log-probability lab passed."),
        ("labs/03_dpo_vertical_slice.py", "DPO lab passed."),
        ("labs/04_group_advantages.py", "Group-advantage lab passed."),
        ("labs/05_synthetic_arithmetic.py", "Synthetic arithmetic lab passed."),
        ("labs/06_exact_generation.py", "Exact generation lab passed."),
    ],
)
def test_cpu_lab_runs_from_repository_root(
    lab: str,
    success_message: str,
    project_root: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, lab],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert success_message in completed.stdout
