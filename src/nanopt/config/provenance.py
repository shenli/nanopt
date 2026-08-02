"""Leaf-level configuration provenance utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProvenanceEntry:
    """Describe the source that most recently assigned a resolved leaf value."""

    source: str
    source_path: str

    def as_dict(self) -> dict[str, str]:
        return {"source": self.source, "source_path": self.source_path}


ProvenanceMap = dict[str, ProvenanceEntry]


def record_leaves(
    value: Any,
    *,
    prefix: str,
    source: str,
    source_prefix: str = "",
) -> ProvenanceMap:
    """Return provenance entries for every scalar or list leaf in ``value``."""

    entries: ProvenanceMap = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            original_path = f"{source_prefix}.{key}" if source_prefix else str(key)
            entries.update(
                record_leaves(
                    child,
                    prefix=path,
                    source=source,
                    source_prefix=original_path,
                )
            )
    else:
        entries[prefix] = ProvenanceEntry(source=source, source_path=source_prefix)
    return entries


def serialize_provenance(provenance: ProvenanceMap) -> dict[str, dict[str, str]]:
    """Create a stable serializable map sorted by resolved path."""

    return {path: provenance[path].as_dict() for path in sorted(provenance)}
