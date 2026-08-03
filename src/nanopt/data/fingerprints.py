"""Canonical hashes for task leakage checks and dataset reproducibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from nanopt.data.arithmetic import ArithmeticGeneratorConfig
from nanopt.data.schemas import ArithmeticTask


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def canonical_task_hash(task: ArithmeticTask) -> str:
    """Hash family, difficulty, and canonical AST before any prompt rendering."""

    value = {
        "family": task.family,
        "difficulty": task.difficulty,
        "canonical_ast": task.canonical_ast.model_dump(mode="json", exclude_none=True),
    }
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def dataset_fingerprint(
    tasks: Sequence[ArithmeticTask],
    *,
    generator_config: ArithmeticGeneratorConfig,
    parser_version: str = "1",
    verifier_version: str = "1",
) -> str:
    """Hash generator inputs, parser/verifier versions, and complete sorted task records."""

    if not tasks:
        raise ValueError("cannot fingerprint an empty dataset")
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("dataset contains duplicate task IDs")
    records = [
        task.model_dump(mode="json", exclude_none=True)
        for task in sorted(tasks, key=lambda item: item.task_id)
    ]
    payload = {
        "schema_version": 1,
        "generator_config": generator_config.model_dump(mode="json"),
        "parser_version": parser_version,
        "verifier_version": verifier_version,
        "records": records,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def fingerprint_records(records: Sequence[dict[str, Any]], *, namespace: str) -> str:
    """Hash an ordered collection of JSON-compatible records under a format namespace.

    The caller controls record ordering because order is part of generated-dataset lineage. A
    namespace prevents structurally similar record types from accidentally sharing an identity.
    """

    if not namespace:
        raise ValueError("fingerprint namespace must not be empty")
    if not records:
        raise ValueError("cannot fingerprint an empty record collection")
    return hashlib.sha256(
        _canonical_bytes({"namespace": namespace, "records": list(records)})
    ).hexdigest()
