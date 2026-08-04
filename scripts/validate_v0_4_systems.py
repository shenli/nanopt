"""Validate the deterministic v0.4 resumable-rollout systems evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from nanopt.runtime.artifacts import read_jsonl, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def validate_v0_4_systems(run_dir: Path, project_root: Path) -> dict[str, Any]:
    """Return compact evidence after validating every v0.4 control-plane contract."""

    run_dir = run_dir.resolve()
    project_root = project_root.resolve()
    manifest = _read_object(run_dir / "run_manifest.json")
    run_schema = _read_object(project_root / "specs/schemas/run_manifest.schema.json")
    jsonschema.Draft202012Validator(run_schema).validate(manifest)
    _require(manifest["stage"] == "systems_lab", "systems evidence stage differs")
    _require(manifest["status"] == "completed", "systems simulation did not complete")

    summary = _read_object(run_dir / "summary.json")
    summary_schema = _read_object(
        project_root / "specs/schemas/systems_simulation_summary.schema.json"
    )
    jsonschema.Draft202012Validator(summary_schema).validate(summary)
    _require(summary["measured_throughput_claim"] is False, "simulation claimed throughput")
    _require(
        summary["simulated_experience_used_for_update"] is False,
        "simulated experience entered a model update",
    )

    targets_value = yaml.safe_load(
        (project_root / "configs/reference_targets.yaml").read_text(encoding="utf-8")
    )
    targets = targets_value["v0_4"]
    comparisons = {item["sync_mode"]: item for item in summary["comparisons"]}
    _require(
        set(comparisons) == {"episode_boundary", "action_boundary"},
        "weight synchronization comparison differs",
    )
    for comparison in comparisons.values():
        _require(comparison["ticks"] == targets["expected_ticks"], "tick count differs")
        _require(
            comparison["policy_updates"] == targets["expected_policy_updates"],
            "policy update count differs",
        )
        _require(
            comparison["partial_checkpoints"] == targets["expected_partial_checkpoints"],
            "partial checkpoint count differs",
        )
        _require(comparison["used_for_model_update"] is False, "comparison entered training")

    for mode in ("episode_boundary", "action_boundary"):
        actual = comparisons[mode]
        expected = targets[mode]
        for field in (
            "mixed_policy_trajectories",
            "stale_trajectories",
            "external_cache_hits",
            "external_cache_misses",
            "recomputed_prompt_tokens",
        ):
            _require(actual[field] == expected[f"expected_{field}"], f"{mode} {field} differs")

    checkpoints = read_jsonl(run_dir / "partial_checkpoints.jsonl")
    checkpoint_schema = _read_object(
        project_root / "specs/schemas/partial_rollout_checkpoint.schema.json"
    )
    checkpoint_validator = jsonschema.Draft202012Validator(checkpoint_schema)
    for checkpoint in checkpoints:
        checkpoint_validator.validate(checkpoint)
    _require(
        len(checkpoints) == 2 * targets["expected_partial_checkpoints"],
        "retained checkpoint record count differs",
    )

    actions = read_jsonl(run_dir / "actions.jsonl")
    decisions = read_jsonl(run_dir / "admission_decisions.jsonl")
    sync_events = read_jsonl(run_dir / "weight_sync_events.jsonl")
    expected_actions = 2 * sum(targets["rollout_action_lengths"])
    _require(len(actions) == expected_actions, "synthetic action record count differs")
    _require(len(decisions) == 2 * len(targets["rollout_action_lengths"]), "decision count differs")
    _require(len(sync_events) == len(checkpoints), "weight sync event count differs")
    _require(
        all(
            record["strict_episode_eligible"] is False
            for record in decisions
            if record["trajectory_id"] == "trajectory-1"
        ),
        "unsafe long trajectory was admitted",
    )

    artifacts = {
        filename: sha256_file(run_dir / filename)
        for filename in (
            "actions.jsonl",
            "partial_checkpoints.jsonl",
            "weight_sync_events.jsonl",
            "admission_decisions.jsonl",
            "summary.json",
            "report.md",
            "run_manifest.json",
        )
    }
    return {
        "schema_version": 1,
        "status": "v0_4_systems_simulation_passed",
        "git_commit": collect_git_metadata(project_root)["commit"],
        "backend": summary["backend"],
        "comparisons": summary["comparisons"],
        "simulated_experience_used_for_update": False,
        "measured_throughput_claim": False,
        "records": {
            "actions": len(actions),
            "partial_checkpoints": len(checkpoints),
            "weight_sync_events": len(sync_events),
            "admission_decisions": len(decisions),
        },
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = validate_v0_4_systems(args.run_dir, args.project_root)
    if args.output:
        write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
