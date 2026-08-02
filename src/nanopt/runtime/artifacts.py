"""Small atomic writers used by all NanoPT run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing and file output."""

    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest used in artifact manifests."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large checkpoints are never loaded into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    """Write beside the destination, flush it, then atomically replace the old file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # The temporary file shares the destination directory, so os.replace stays on one
        # filesystem and provides the atomic replacement guarantee we need for manifests.
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON document with stable formatting."""

    _atomic_write(path, canonical_json(value))


def write_yaml(path: Path, value: Any) -> None:
    """Atomically replace a YAML document with sorted, stable keys."""

    content = yaml.safe_dump(
        value,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).encode()
    _atomic_write(path, content)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Append one complete JSONL record with a single operating-system write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    encoded = line.encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        # One os.write keeps a record together when several writers append to the same file.
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short JSONL write: wrote {written} of {len(encoded)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL and identify the exact malformed line after an interrupted write."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(value)
    return records
