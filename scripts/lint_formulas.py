"""Reject unsupported raw MathJax delimiters in authored Markdown prose."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FORBIDDEN = (r"\(", r"\)", r"\[", r"\]")
INLINE_CODE = re.compile(r"`[^`]*`")


def lint_file(path: Path) -> list[str]:
    errors: list[str] = []
    fence: str | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        prose = INLINE_CODE.sub("", line)
        for delimiter in FORBIDDEN:
            if delimiter in prose:
                errors.append(f"{path}:{line_number}: unsupported formula delimiter {delimiter!r}")
    if fence is not None:
        errors.append(f"{path}: unclosed Markdown code fence")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()
    files: list[Path] = []
    for path in arguments.paths:
        files.extend(sorted(path.rglob("*.md")) if path.is_dir() else [path])
    errors = [error for path in files for error in lint_file(path)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Formula lint passed for {len(files)} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
