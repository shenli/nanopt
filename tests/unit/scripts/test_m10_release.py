from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_m10_release import validate_release


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
