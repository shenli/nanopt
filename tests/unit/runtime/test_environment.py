from __future__ import annotations

from pathlib import Path

from nanopt.runtime.environment import collect_environment, collect_git_metadata


def test_environment_omits_identity_and_environment_values() -> None:
    report = collect_environment()
    assert "hostname" not in report
    assert "username" not in report
    assert set(report) == {
        "schema_version",
        "os",
        "os_release",
        "architecture",
        "python",
        "packages",
    }


def test_git_metadata_has_reproducibility_fields(project_root: Path) -> None:
    metadata = collect_git_metadata(project_root)
    assert len(metadata["commit"]) == 40
    assert isinstance(metadata["dirty"], bool)
    assert "@github.com" not in (metadata["remote"] or "")
