"""Validate the frozen v0.1 release contract and optional distribution archives."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from nanopt.runtime.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata

RELEASE_VERSION = "0.1.0"
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_TRACKED_NAMES = {
    "AGENT_START_HERE.md",
    "NANOPT_MASTER_PLAN.md",
    "PACKAGE_INVENTORY.md",
    "PROMPT_FOR_CODING_AGENT.md",
    "SHA256SUMS.txt",
    "WEB_RESEARCH_SNAPSHOT.md",
}
PRIVATE_TRACKED_PREFIXES = (
    "artifacts/cache/",
    "artifacts/pipelines/",
    "artifacts/runs/",
    "artifacts/tmp/",
    "repo_seed/",
    "source_materials/",
)
PERSONAL_PATH_PATTERNS = (
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\]+\\"),
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
NON_ENGLISH_SCRIPT = re.compile(
    "[\u0400-\u052f\u0590-\u05ff\u0600-\u06ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"
)
REDACTION_FIXTURE = "tests/unit/reporting/test_builder.py"


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


def _tracked_files(project_root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return sorted(path for path in completed.stdout.decode().split("\0") if path)


def _scan_public_tree(project_root: Path) -> dict[str, Any]:
    """Reject private inputs, personal paths, secrets, and non-English scripts."""

    tracked = _tracked_files(project_root)
    _require(not (PRIVATE_TRACKED_NAMES & set(tracked)), "private handoff file is tracked")
    _require(
        not any(path.startswith(PRIVATE_TRACKED_PREFIXES) for path in tracked),
        "private or generated artifact directory is tracked",
    )
    _require(
        not any(path.startswith(".github/workflows/") for path in tracked),
        "GitHub Actions workflows are outside the local-validation policy",
    )

    text_files = 0
    total_bytes = 0
    file_hashes: dict[str, str] = {}
    for relative in tracked:
        path = project_root / relative
        _require(path.is_file(), f"tracked path is not a regular file: {relative}")
        size = path.stat().st_size
        _require(size <= 2 * 1024 * 1024, f"tracked file exceeds 2 MiB: {relative}")
        total_bytes += size
        file_hashes[relative] = sha256_file(path)
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE"}:
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        text = raw.decode("utf-8")
        text_files += 1
        if relative == REDACTION_FIXTURE:
            # These exact fake values exercise the report redactor. Removing only the reviewed
            # fixtures means any new path or token in this test still fails the public-tree scan.
            text = text.replace("/Users/private", "<redaction-fixture>")
            fake_github_token = "gho_" + "abcdefghijklmnopqrstuvwxyz"
            text = text.replace(fake_github_token, "<redaction-fixture>")
        for pattern in (*PERSONAL_PATH_PATTERNS, *SECRET_PATTERNS):
            _require(pattern.search(text) is None, f"public-content audit failed: {relative}")
        _require(
            NON_ENGLISH_SCRIPT.search(text) is None,
            f"non-English writing system found in public content: {relative}",
        )

    return {
        "tracked_files": len(tracked),
        "text_files_scanned": text_files,
        "total_bytes": total_bytes,
        "tree_manifest_sha256": sha256_bytes(canonical_json(file_hashes)),
        "redaction_fixture_exception": REDACTION_FIXTURE,
        "github_actions_workflows": 0,
    }


def _project_versions(project_root: Path) -> dict[str, str]:
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((project_root / "uv.lock").read_text(encoding="utf-8"))
    citation = _load_yaml(project_root / "CITATION.cff")
    version_text = (project_root / "src/nanopt/version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', version_text, flags=re.MULTILINE)
    _require(match is not None, "source version is missing")
    lock_versions = [
        package["version"] for package in lock["package"] if package["name"] == "nanopt"
    ]
    _require(len(lock_versions) == 1, "uv.lock must contain one nanopt package")
    versions = {
        "pyproject": pyproject["project"]["version"],
        "source": match.group(1),
        "lock": lock_versions[0],
        "citation": str(citation["version"]),
    }
    _require(set(versions.values()) == {RELEASE_VERSION}, f"release versions disagree: {versions}")
    _require(pyproject["project"]["license"] == "Apache-2.0", "package license changed")
    _require(citation["license"] == "Apache-2.0", "citation license changed")
    return versions


def _validate_model_and_references(project_root: Path, release: dict[str, Any]) -> dict[str, Any]:
    model = _load_yaml(project_root / "configs/models/qwen3_0_6b_base.yaml")
    source = model["source"]
    for field in ("model_id", "revision", "tokenizer_revision"):
        _require(source[field] == release["model"][field], f"release model {field} disagrees")
    _require(source["trust_remote_code"] is False, "reference model may not trust remote code")
    _require(model["loading"]["use_safetensors"] is True, "reference model must use safetensors")

    retained: dict[str, dict[str, str]] = {}
    for key in (
        "retained_pipeline_evidence",
        "retained_agent_evidence",
        "retained_curriculum_evidence",
    ):
        relative = release["reference"][key]
        path = (project_root / relative).resolve()
        _require(
            path.is_relative_to(project_root), f"reference evidence escapes project: {relative}"
        )
        evidence = _load_json(path)
        status = evidence.get("status")
        _require(
            isinstance(status, str) and status.endswith("_passed"),
            f"retained evidence did not pass: {relative}",
        )
        retained[relative] = {"status": status, "sha256": sha256_file(path)}
    return {
        "model_profile_sha256": sha256_file(project_root / "configs/models/qwen3_0_6b_base.yaml"),
        "model_revision": source["revision"],
        "tokenizer_revision": source["tokenizer_revision"],
        "model_license": release["model"]["license"],
        "retained_evidence": retained,
    }


def _installed_dependency_metadata() -> dict[str, Any]:
    records: list[dict[str, str]] = []
    missing_license_metadata: list[str] = []
    for distribution in sorted(
        importlib.metadata.distributions(),
        key=lambda item: (item.metadata.get("Name") or "").lower(),
    ):
        name = distribution.metadata.get("Name")
        if not name or name == "nanopt":
            continue
        license_value = (
            distribution.metadata.get("License-Expression")
            or distribution.metadata.get("License")
            or ""
        ).strip()
        if not license_value or license_value.upper() == "UNKNOWN":
            missing_license_metadata.append(f"{name}=={distribution.version}")
        records.append(
            {"name": name, "version": distribution.version, "license_metadata": license_value}
        )
    return {
        "installed_distributions": len(records),
        "records": records,
        "missing_or_unknown_license_metadata": missing_license_metadata,
        "manual_review": "docs/reference/dependency-license-audit.md",
    }


def _archive_members(path: Path) -> tuple[list[str], bytes]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            metadata_name = next(name for name in members if name.endswith(".dist-info/METADATA"))
            return members, archive.read(metadata_name)
    with tarfile.open(path, mode="r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
        metadata_member = next(member for member in members if member.endswith("/PKG-INFO"))
        extracted = archive.extractfile(metadata_member)
        _require(extracted is not None, f"could not read metadata: {path}")
        return members, extracted.read()


def _validate_distributions(dist_dir: Path) -> dict[str, Any]:
    expected = {
        f"nanopt-{RELEASE_VERSION}-py3-none-any.whl",
        f"nanopt-{RELEASE_VERSION}.tar.gz",
    }
    found = {path.name for path in dist_dir.iterdir() if path.is_file()}
    _require(found == expected, f"unexpected distribution files: {sorted(found)}")
    artifacts: dict[str, dict[str, Any]] = {}
    for filename in sorted(found):
        path = dist_dir / filename
        members, metadata = _archive_members(path)
        _require(b"Name: nanopt\n" in metadata, f"archive name metadata failed: {filename}")
        _require(
            f"Version: {RELEASE_VERSION}\n".encode() in metadata,
            f"archive version metadata failed: {filename}",
        )
        _require(any("LICENSE" in member for member in members), f"license absent: {filename}")
        _require(
            not any(
                private in member
                for member in members
                for private in (*PRIVATE_TRACKED_NAMES, "source_materials")
            ),
            f"private content found in archive: {filename}",
        )
        artifacts[filename] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "files": len(members),
        }
    return artifacts


def validate_release(project_root: Path, *, dist_dir: Path | None) -> dict[str, Any]:
    """Return structural or built-release evidence for the frozen v0.1 candidate."""

    project_root = project_root.resolve()
    release_path = project_root / "configs/releases/v0_1_0.yaml"
    release = _load_yaml(release_path)
    schema = _load_json(project_root / "specs/schemas/release.schema.json")
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        release
    )
    versions = _project_versions(project_root)
    public_tree = _scan_public_tree(project_root)
    model_and_references = _validate_model_and_references(project_root, release)

    license_text = (project_root / "LICENSE").read_text(encoding="utf-8")
    _require("Apache License" in license_text and "Version 2.0" in license_text, "invalid LICENSE")
    _require((project_root / "docs/data/dataset-card.md").is_file(), "dataset card is missing")
    _require(
        (project_root / "docs/reference/dependency-license-audit.md").is_file(),
        "dependency license audit is missing",
    )

    git = collect_git_metadata(project_root)
    distributions: dict[str, Any] = {}
    if dist_dir is not None:
        _require(git["dirty"] is False, "built release evidence requires a clean checkout")
        distributions = _validate_distributions(dist_dir.resolve())

    return {
        "schema_version": 1,
        "status": "m10_release_passed" if dist_dir is not None else "m10_release_structure_passed",
        "release": {
            "version": release["version"],
            "tag": release["tag"],
            "git_commit": git["commit"],
            "manifest_sha256": sha256_file(release_path),
            "uv_lock_sha256": sha256_file(project_root / "uv.lock"),
            "reference_targets_sha256": sha256_file(
                project_root / "configs/reference_targets.yaml"
            ),
        },
        "versions": versions,
        "public_tree": public_tree,
        "supply_chain": {
            **model_and_references,
            "dependencies": _installed_dependency_metadata(),
        },
        "distributions": distributions,
        "publication": release["publication"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = validate_release(args.project_root, dist_dir=args.dist_dir)
    if args.output:
        write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
