"""Validate the 20-chapter curriculum and optionally execute every local lab."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from nanopt.runtime.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata

LOCAL_TIERS = {"cpu", "systems_simulation"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected YAML object: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _safe_project_path(project_root: Path, relative: str) -> Path:
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts, f"unsafe manifest path: {relative}")
    resolved = (project_root / path).resolve()
    _require(resolved.is_relative_to(project_root), f"manifest path escapes project: {relative}")
    return resolved


def _lab_script(command: str) -> str:
    parts = shlex.split(command)
    _require(
        len(parts) == 4 and parts[:3] == ["uv", "run", "python"],
        f"local lab must use 'uv run python LAB': {command}",
    )
    _require(parts[3].startswith("labs/") and parts[3].endswith(".py"), f"invalid lab: {command}")
    return parts[3]


def validate_curriculum(
    project_root: Path,
    *,
    execute_labs: bool,
) -> dict[str, Any]:
    """Return public-safe structural, execution, and prior-reference evidence."""

    project_root = project_root.resolve()
    manifest_path = project_root / "specs/curriculum.yaml"
    manifest = _load_yaml(manifest_path)
    schema = _load_json(project_root / "specs/schemas/curriculum.schema.json")
    jsonschema.Draft202012Validator(schema).validate(manifest)

    chapters = manifest["chapters"]
    _require(
        [chapter["id"] for chapter in chapters] == list(range(20)), "chapter IDs must be 0..19"
    )
    entries = [manifest["prerequisite"], *chapters]
    mkdocs = (project_root / "mkdocs.yml").read_text(encoding="utf-8")
    for entry in entries:
        doc = _safe_project_path(project_root, entry["path"])
        _require(doc.is_file(), f"chapter is missing: {entry['path']}")
        content = doc.read_text(encoding="utf-8")
        _require(content.startswith("# "), f"chapter lacks one top-level title: {entry['path']}")
        _require(
            "## Learning objectives" in content or "## Learning goals" in content,
            f"chapter lacks learning objectives: {entry['path']}",
        )
        nav_path = entry["path"].removeprefix("docs/")
        _require(nav_path in mkdocs, f"chapter is absent from MkDocs navigation: {entry['path']}")

    declared_local: dict[str, str] = {}
    reference_evidence: dict[str, dict[str, str]] = {}
    declared_count = 0
    reference_declarations = 0
    for entry in entries:
        for lab in entry["labs"]:
            declared_count += 1
            tier = lab["tier"]
            command = lab["command"]
            if tier in LOCAL_TIERS:
                script = _lab_script(command)
                _require(
                    _safe_project_path(project_root, script).is_file(), f"lab is missing: {script}"
                )
                previous_tier = declared_local.get(command)
                _require(
                    previous_tier is None or previous_tier == tier,
                    f"lab command has conflicting tiers: {command}",
                )
                declared_local[command] = tier
            elif tier == "reference":
                reference_declarations += 1
                command_parts = shlex.split(command)
                _require(
                    len(command_parts) == 1, f"reference command must be one script: {command}"
                )
                _require(
                    _safe_project_path(project_root, command_parts[0]).is_file(),
                    f"reference script is missing: {command}",
                )
                evidence_relative = lab["evidence"]
                evidence_path = _safe_project_path(project_root, evidence_relative)
                evidence = _load_json(evidence_path)
                status = evidence.get("status")
                _require(
                    isinstance(status, str) and status.endswith("_passed"),
                    f"reference evidence did not pass: {evidence_relative}",
                )
                reference_evidence[evidence_relative] = {
                    "status": status,
                    "sha256": sha256_file(evidence_path),
                }

    all_lab_files = {
        path.relative_to(project_root).as_posix() for path in (project_root / "labs").glob("*.py")
    }
    declared_lab_files = {_lab_script(command) for command in declared_local}
    _require(
        all_lab_files == declared_lab_files,
        "every checked-in lab must be declared exactly by the curriculum; "
        f"undeclared={sorted(all_lab_files - declared_lab_files)}, "
        f"missing={sorted(declared_lab_files - all_lab_files)}",
    )

    executions: list[dict[str, str | float]] = []
    if execute_labs:
        for command, tier in declared_local.items():
            script = _lab_script(command)
            started = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, script],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            executions.append(
                {
                    "command": command,
                    "tier": tier,
                    "duration_seconds": time.perf_counter() - started,
                    "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
                    "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
                }
            )

    git = collect_git_metadata(project_root)
    if execute_labs:
        _require(git["dirty"] is False, "curriculum evidence requires a clean checkout")
    course_fingerprint = sha256_bytes(
        canonical_json(
            {
                "manifest": manifest,
                "chapters": {
                    entry["path"]: sha256_file(_safe_project_path(project_root, entry["path"]))
                    for entry in entries
                },
                "labs": {
                    script: sha256_file(_safe_project_path(project_root, script))
                    for script in sorted(all_lab_files)
                },
            }
        )
    )
    return {
        "schema_version": 1,
        "status": "m9_curriculum_passed" if execute_labs else "m9_curriculum_structure_passed",
        "git_commit": git["commit"],
        "course": {
            "prerequisite_chapters": 1,
            "numbered_chapters": len(chapters),
            "course_fingerprint": course_fingerprint,
        },
        "labs": {
            "declared_uses": declared_count,
            "unique_local_labs": len(declared_local),
            "executed_local_labs": len(executions),
            "cpu_labs": sum(tier == "cpu" for tier in declared_local.values()),
            "systems_simulations": sum(
                tier == "systems_simulation" for tier in declared_local.values()
            ),
            "reference_declarations": reference_declarations,
            "executions": executions,
        },
        "reference_evidence": reference_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute-labs", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = validate_curriculum(args.project_root, execute_labs=args.execute_labs)
    if args.output:
        write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
