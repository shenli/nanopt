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
        ("labs/07_completion_only_sft.py", "Completion-only SFT lab passed."),
        ("labs/08_controlled_preferences.py", "Controlled preference lab passed."),
        ("labs/09_exact_rlvr_trajectory.py", "Exact RLVR trajectory lab passed."),
        ("labs/10_mini_swe_environment.py", "MiniSWE reset and semantic replay lab passed."),
        ("labs/11_reward_ranking.py", "Reward-ranking lab passed."),
        ("labs/12_reinforce.py", "REINFORCE lab passed."),
        ("labs/13_ppo_clipping.py", "PPO-clipping lab passed."),
        ("labs/14_reward_hacking.py", "Reward-hacking lab passed."),
        ("labs/15_rollout_scheduler.py", "Rollout-scheduler simulation passed."),
        ("labs/16_production_flywheel.py", "Production-flywheel simulation passed."),
        ("labs/17_task_card.py", "Task-card lab passed."),
        ("labs/18_artifact_lineage.py", "Artifact-lineage lab passed."),
        ("labs/19_optimizer_step.py", "Optimizer-step lab passed."),
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
        timeout=60,
    )

    assert success_message in completed.stdout
