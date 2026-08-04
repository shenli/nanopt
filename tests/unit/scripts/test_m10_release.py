from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.validate_m10_release import _scan_public_tree, validate_release


def test_frozen_release_contract_and_public_tree_pass(project_root: Path) -> None:
    evidence = validate_release(project_root, dist_dir=None)

    assert evidence["status"] == "m10_release_structure_passed"
    assert evidence["release"]["version"] == "0.3.0"
    assert evidence["versions"] == {
        "pyproject": "0.3.0",
        "source": "0.3.0",
        "lock": "0.3.0",
        "citation": "0.3.0",
    }
    assert evidence["public_tree"]["github_actions_workflows"] == 0
    assert evidence["public_tree"]["candidate_files"] >= evidence["public_tree"]["tracked_files"]
    assert evidence["supply_chain"]["model_revision"] == (
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    )
    assert len(evidence["supply_chain"]["retained_evidence"]) == 5

    agent_rl_evidence = json.loads(
        (project_root / "docs/reference/evidence/v0.3-agent-rl-85ca98b.json").read_text()
    )
    assert agent_rl_evidence["git_commit"].startswith("85ca98b")
    assert agent_rl_evidence["selection"] == {
        "final_training_policy_version": 4,
        "parent_policy_selectable": False,
        "selected_policy_version": 1,
        "selection_rule": "highest_post_update_validation_reward_then_earliest",
        "validation_rewards_by_policy_version": [1.0, 1.0, 0.2333333333333333, 1.0, 0.0],
    }


def test_public_tree_scan_includes_untracked_candidates(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.md").write_text("# Public\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.md"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.md").write_text(
        "accidental path: /" + "Users/private/project\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"untracked\.md"):
        _scan_public_tree(tmp_path)
