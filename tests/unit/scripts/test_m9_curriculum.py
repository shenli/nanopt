from __future__ import annotations

from pathlib import Path

from scripts.validate_m9_curriculum import validate_curriculum


def test_complete_curriculum_manifest_has_every_chapter_and_lab(project_root: Path) -> None:
    evidence = validate_curriculum(project_root, execute_labs=False)

    assert evidence["status"] == "m9_curriculum_structure_passed"
    assert evidence["course"]["numbered_chapters"] == 21
    assert evidence["course"]["prerequisite_chapters"] == 1
    assert evidence["labs"]["unique_local_labs"] == 21
    assert evidence["labs"]["systems_simulations"] == 2
    assert evidence["labs"]["reference_declarations"] == 7
