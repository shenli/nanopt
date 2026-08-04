from __future__ import annotations

from pathlib import Path

from scripts.validate_m10_release import validate_release


def test_frozen_release_contract_and_public_tree_pass(project_root: Path) -> None:
    evidence = validate_release(project_root, dist_dir=None)

    assert evidence["status"] == "m10_release_structure_passed"
    assert evidence["release"]["version"] == "0.2.0"
    assert evidence["versions"] == {
        "pyproject": "0.2.0",
        "source": "0.2.0",
        "lock": "0.2.0",
        "citation": "0.2.0",
    }
    assert evidence["public_tree"]["github_actions_workflows"] == 0
    assert evidence["supply_chain"]["model_revision"] == (
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    )
    assert len(evidence["supply_chain"]["retained_evidence"]) == 4
