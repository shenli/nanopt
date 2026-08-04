from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from nanopt.config.loader import ConfigRepository
from nanopt.config.resolver import resolve_config
from nanopt.systems.run import execute_systems_lab_run
from scripts.validate_v0_4_systems import validate_v0_4_systems


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_systems_lab_writes_inspectable_non_training_artifacts(
    tmp_path: Path,
    project_root: Path,
) -> None:
    resolved = resolve_config(
        repository=ConfigRepository(project_root / "configs"),
        hardware_id="rtx_4070_ti_super_16gb",
        model_id="qwen3_0_6b_base",
        experiment_id="resumable_rollouts",
    )
    context = execute_systems_lab_run(
        resolved,
        artifacts_root=tmp_path,
        run_id="systems-fixture",
    )

    summary = json.loads((context.run_dir / "summary.json").read_text())
    assert summary["status"] == "v0_4_systems_simulation_passed"
    assert summary["measured_throughput_claim"] is False
    assert summary["simulated_experience_used_for_update"] is False
    assert {item["sync_mode"] for item in summary["comparisons"]} == {
        "episode_boundary",
        "action_boundary",
    }
    for filename in (
        "actions.jsonl",
        "partial_checkpoints.jsonl",
        "weight_sync_events.jsonl",
        "admission_decisions.jsonl",
        "report.md",
    ):
        assert (context.run_dir / filename).is_file()

    manifest = json.loads(context.manifest_path.read_text())
    schema = json.loads((project_root / "specs/schemas/run_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert manifest["stage"] == "systems_lab"
    assert manifest["status"] == "completed"

    summary_schema = json.loads(
        (project_root / "specs/schemas/systems_simulation_summary.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(summary_schema).validate(summary)
    checkpoint_schema = json.loads(
        (project_root / "specs/schemas/partial_rollout_checkpoint.schema.json").read_text()
    )
    checkpoint_validator = jsonschema.Draft202012Validator(checkpoint_schema)
    for checkpoint in _jsonl(context.run_dir / "partial_checkpoints.jsonl"):
        checkpoint_validator.validate(checkpoint)

    evidence = validate_v0_4_systems(context.run_dir, project_root)
    assert evidence["status"] == "v0_4_systems_simulation_passed"
    assert evidence["records"] == {
        "actions": 28,
        "partial_checkpoints": 6,
        "weight_sync_events": 6,
        "admission_decisions": 8,
    }
