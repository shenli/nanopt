from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema

from nanopt.config.loader import ConfigRepository
from nanopt.config.resolver import resolve_config
from nanopt.runtime.run_context import create_run_context, make_run_id


def test_run_id_is_stable_and_contains_no_identity() -> None:
    now = datetime(2026, 8, 2, 1, 2, 3, tzinfo=UTC)
    first = make_run_id("SFT", {"seed": 42}, now)
    second = make_run_id("SFT", {"seed": 42}, now)
    assert first == second
    assert first.startswith("20260802-010203_sft_")
    assert "/" not in first


def test_run_context_matches_manifest_schema_and_persists_failure(
    tmp_path: Path, project_root: Path
) -> None:
    repository = ConfigRepository(project_root / "configs")
    result = resolve_config(
        repository=repository,
        hardware_id="rtx_4070_ti_super_16gb",
        model_id="qwen3_0_6b_base",
        experiment_id="math_sft",
    )
    context = create_run_context(
        result,
        artifacts_root=tmp_path / "runs",
        run_id="fixture_run",
        git_root=project_root,
    )
    context.set_status("running")
    context.set_status(
        "failed",
        failure={
            "type": "RuntimeError",
            "message": "fixture",
            "phase": "test",
            "traceback_file": None,
        },
    )

    manifest = json.loads(context.manifest_path.read_text())
    schema = json.loads((project_root / "specs/schemas/run_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        manifest
    )
    assert manifest["status"] == "failed"
    assert manifest["started_at"] is not None
    assert manifest["finished_at"] is not None
    assert manifest["failure"]["message"] == "fixture"
    assert (context.run_dir / "resolved_config.yaml").is_file()
    assert (context.run_dir / "config_provenance.yaml").is_file()
    assert (context.run_dir / "environment.json").is_file()
    assert (context.run_dir / "metrics.jsonl").read_bytes() == b""
