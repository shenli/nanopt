"""Write deterministic checksums for the retained files in one reference evidence bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from nanopt.runtime.artifacts import sha256_file, write_json


def write_checksums(evidence_root: Path, output: Path) -> dict[str, str]:
    """Hash evidence files while excluding the disposable fresh environment and live log."""

    excluded_roots = {"fresh-venv"}
    excluded_files = {
        output.name,
        "commands.log",
        "m7_pipeline_evidence.json",
        "m8_agent_evidence.json",
        "v0.2-agent-sft-evidence.json",
    }
    checksums = {
        path.relative_to(evidence_root).as_posix(): sha256_file(path)
        for path in sorted(evidence_root.rglob("*"))
        if path.is_file()
        and path.relative_to(evidence_root).parts[0] not in excluded_roots
        and path.name not in excluded_files
    }
    write_json(output, checksums)
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_checksums(args.evidence_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
