"""Sanitized environment and source-control metadata capture."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _sanitize_remote(remote: str | None) -> str | None:
    if not remote:
        return None
    if re.match(r"^[^/@:]+@[^:]+:", remote):
        return re.sub(r"^[^/@:]+@", "", remote)
    parts = urlsplit(remote)
    if parts.scheme and parts.hostname:
        hostname = parts.hostname
        if parts.port:
            hostname = f"{hostname}:{parts.port}"
        return urlunsplit((parts.scheme, hostname, parts.path, parts.query, parts.fragment))
    return remote


def collect_git_metadata(cwd: Path | None = None) -> dict[str, Any]:
    """Collect reproducibility metadata without storing local paths or credentials."""

    root = cwd or Path.cwd()
    commit = _git(["rev-parse", "HEAD"], root) or "unavailable"
    status = _git(["status", "--porcelain"], root)
    tag = _git(["describe", "--tags", "--exact-match"], root)
    remote = _sanitize_remote(_git(["remote", "get-url", "origin"], root))
    return {"commit": commit, "dirty": bool(status), "tag": tag, "remote": remote}


def collect_environment() -> dict[str, Any]:
    """Capture stable software facts; omit host names, user names, and environment values."""

    packages = ["nanopt", "torch", "transformers", "peft", "pydantic", "typer", "PyYAML"]
    return {
        "schema_version": 1,
        "os": platform.system().lower(),
        "os_release": platform.release(),
        "architecture": platform.machine().lower(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "packages": {package: package_version(package) for package in packages},
    }
