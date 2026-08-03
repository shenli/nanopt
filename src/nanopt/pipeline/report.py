"""Build the compact Base -> SFT -> DPO -> GRPO comparison report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from nanopt.runtime.artifacts import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_text,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _repeat_identity(path: Path) -> list[dict[str, Any]]:
    """Keep generation evidence while excluding run identity and measured wall time."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON objects in {path}")
            records.append(
                {
                    key: item
                    for key, item in value.items()
                    if key not in {"result_id", "run_id", "generation_seconds"}
                }
            )
    return records


def build_pipeline_report(
    pipeline_dir: Path,
    *,
    evaluations: dict[str, Path],
    training_runs: dict[str, Path],
    checkpoint_hashes: dict[str, str],
    repeat_evaluation: Path,
) -> dict[str, str]:
    """Persist machine-readable and human-readable comparisons from saved artifacts only."""

    rows: list[dict[str, Any]] = []
    for checkpoint, run_dir in evaluations.items():
        summary = _read_json(run_dir / "summary.json")
        accuracy = summary["accuracy"]
        parse_rate = summary["parse_rate"]
        rows.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_hashes[checkpoint],
                "examples": summary["examples"],
                "accuracy": accuracy["estimate"],
                "accuracy_lower": accuracy["lower"],
                "accuracy_upper": accuracy["upper"],
                "parse_rate": parse_rate["estimate"],
                "evaluation_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            }
        )

    training: dict[str, Any] = {}
    for stage, run_dir in training_runs.items():
        summary = _read_json(run_dir / "summary.json")
        training[stage] = {
            "run_id": summary["run_id"],
            "peak_reserved_bytes": summary["peak_reserved_bytes"],
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        }

    final_samples = evaluations["grpo"] / "samples.jsonl"
    repeat_samples = repeat_evaluation / "samples.jsonl"
    final_identity = _repeat_identity(final_samples)
    repeat_identity = _repeat_identity(repeat_samples)
    repeat_exact = final_identity == repeat_identity
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_run_id": pipeline_dir.name,
        "evaluations": rows,
        "training": training,
        "final_evaluation_repeat": {
            "exact_generation_match": repeat_exact,
            "first_samples_sha256": sha256_file(final_samples),
            "repeat_samples_sha256": sha256_file(repeat_samples),
            "normalized_generation_sha256": sha256_bytes(canonical_json(final_identity)),
        },
    }
    write_json(pipeline_dir / "comparison.json", comparison)

    table_rows = "\n".join(
        "| {checkpoint} | `{checkpoint_sha256}` | {accuracy:.2%} | "
        "[{accuracy_lower:.2%}, {accuracy_upper:.2%}] | {parse_rate:.2%} |".format(**row)
        for row in rows
    )
    repeat_label = str(repeat_exact).lower()
    markdown = f"""# NanoPT end-to-end pipeline report

This report is rebuilt only from retained child-run artifacts. Accuracy uses the same frozen
protected tasks, renderer, parser, verifier, and deterministic generation settings at every stage.

| Checkpoint | SHA-256 | Accuracy | 95% Wilson interval | Parse rate |
| --- | --- | ---: | ---: | ---: |
{table_rows}

The repeated final evaluation has an exact generation-evidence match: **{repeat_label}**.
Training memory and complete stage timing are recorded in `comparison.json` and
`pipeline_manifest.json`; failed attempts remain in the parent failure/retry log.
"""
    write_text(pipeline_dir / "report.md", markdown)
    write_text(
        pipeline_dir / "report.html",
        "<!doctype html><html><head><meta charset='utf-8'><title>NanoPT pipeline report"
        "</title></head><body><pre>" + html.escape(markdown) + "</pre></body></html>\n",
    )
    return {
        name: sha256_file(pipeline_dir / name)
        for name in ("comparison.json", "report.md", "report.html")
    }
