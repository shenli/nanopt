"""Validate exact-token data, Agent SFT training, and Docker behavior evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from nanopt.agent.records import AgentRunSummary, AgentTrajectory
from nanopt.agent.sft_data import read_agent_sft_dataset
from nanopt.agent.sft_records import AgentSftSummary
from nanopt.config.loader import ConfigRepository
from nanopt.runtime.artifacts import read_jsonl, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata
from nanopt.sft.checkpoint import sha256_directory
from scripts.validate_m3_reference_smoke import _read_object, _require, _validate_doctor
from scripts.validate_m8_reference_agent import _validate_run_manifest, _verify_checksums


def _validate_agent_run(
    evidence_root: Path,
    project_root: Path,
    name: str,
    *,
    expected_context: str,
) -> tuple[AgentRunSummary, list[AgentTrajectory]]:
    run_dir = evidence_root / "runs" / name
    manifest = _validate_run_manifest(run_dir, project_root, "model")
    summary = AgentRunSummary.model_validate(_read_object(run_dir / "summary.json"), strict=True)
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    _require(
        resolved["experiment"]["policy"]["context_policy"] == expected_context,
        f"{name} context policy differs",
    )
    _require(manifest["agent_environment"]["backend"] == "docker", f"{name} was not Docker")
    trajectories = [
        AgentTrajectory.model_validate(item, strict=True)
        for item in read_jsonl(run_dir / "trajectories.jsonl")
    ]
    _require(len(trajectories) == summary.tasks == 1, f"{name} task count differs")
    _require(
        all(step.model_token_ids is not None for item in trajectories for step in item.steps),
        f"{name} omitted model token IDs",
    )
    return summary, trajectories


def validate_v0_2_agent_sft(evidence_root: Path, project_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    project_root = project_root.resolve()
    doctor = _validate_doctor(evidence_root, project_root)
    _require(doctor["docker"]["daemon_reachable"] is True, "Docker daemon was unreachable")
    checksums = _verify_checksums(evidence_root)

    dataset_manifest, examples = read_agent_sft_dataset(evidence_root / "data")
    manifest_schema = _read_object(project_root / "specs/schemas/agent_sft_dataset.schema.json")
    example_schema = _read_object(project_root / "specs/schemas/agent_sft_example.schema.json")
    jsonschema.Draft202012Validator(manifest_schema).validate(
        dataset_manifest.model_dump(mode="json")
    )
    for example in examples:
        jsonschema.Draft202012Validator(example_schema).validate(example.model_dump(mode="json"))
    _require(dataset_manifest.train_examples == 24, "Agent SFT train-example count differs")
    _require(dataset_manifest.validation_examples == 6, "validation-example count differs")
    _require(dataset_manifest.demonstration_examples == 25, "demonstration count differs")
    _require(dataset_manifest.recovery_examples == 5, "recovery count differs")
    _require(dataset_manifest.exact_replays_passed == 10, "source replays did not all pass")
    _require(
        all(not any(item.action_mask[: item.prompt_length]) for item in examples),
        "a prompt token is active in an action mask",
    )

    train_dir = evidence_root / "runs/agent-sft"
    train_manifest = _read_object(train_dir / "run_manifest.json")
    run_schema = _read_object(project_root / "specs/schemas/run_manifest.schema.json")
    jsonschema.Draft202012Validator(run_schema).validate(train_manifest)
    _require(train_manifest["stage"] == "agent_sft", "training stage differs")
    _require(train_manifest["status"] == "completed", "Agent SFT run did not complete")
    _require(train_manifest["git"]["dirty"] is False, "Agent SFT used a dirty checkout")
    _require(
        train_manifest["git"]["commit"] == collect_git_metadata(project_root)["commit"],
        "Agent SFT commit differs",
    )
    train_summary = AgentSftSummary.model_validate(
        _read_object(train_dir / "summary.json"), strict=True
    )
    summary_schema = _read_object(project_root / "specs/schemas/agent_sft_summary.schema.json")
    jsonschema.Draft202012Validator(summary_schema).validate(train_summary.model_dump(mode="json"))
    _require(
        train_summary.final_validation_nll < train_summary.initial_validation_nll,
        "held-out Agent SFT NLL did not improve",
    )
    _require(
        train_summary.final_validation_token_accuracy >= 0.9,
        "held-out Agent SFT token accuracy is below 90%",
    )
    hardware = ConfigRepository(project_root / "configs").hardware("rtx_4070_ti_super_16gb")
    hard_bytes = int(hardware.memory_budget.hard_peak_reserved_gib * 1024**3)
    _require(train_summary.peak_reserved_bytes <= hard_bytes, "Agent SFT exceeded hard VRAM")
    adapter_relative = train_manifest["checkpoint"]["path"]
    adapter_dir = train_dir / str(adapter_relative)
    _require(
        sha256_directory(adapter_dir) == train_manifest["checkpoint"]["sha256"],
        "final Agent SFT adapter hash differs",
    )

    baseline, _ = _validate_agent_run(
        evidence_root, project_root, "base-train-task", expected_context="full_transcript"
    )
    adapted, adapted_trajectories = _validate_agent_run(
        evidence_root, project_root, "adapted-train-task", expected_context="full_transcript"
    )
    snapshot, _ = _validate_agent_run(
        evidence_root,
        project_root,
        "adapted-train-task-snapshot",
        expected_context="observation_snapshot",
    )
    held_out, held_out_trajectories = _validate_agent_run(
        evidence_root, project_root, "adapted-held-out-task", expected_context="full_transcript"
    )
    _require(adapted.action_validity_rate == 1, "adapted training-task actions were invalid")
    _require(
        adapted.mean_score == 1 and adapted.solved == 1, "adapted training task was not solved"
    )
    _require(
        adapted.action_validity_rate > baseline.action_validity_rate,
        "Agent SFT did not improve action validity over base",
    )
    _require(
        adapted.action_validity_rate > snapshot.action_validity_rate,
        "full transcript did not outperform snapshot validity",
    )
    _require(held_out.action_validity_rate == 1, "held-out actions were not all valid")
    _require(held_out.policy_violations == 0, "held-out rollout had policy violations")
    _require(adapted_trajectories[0].task_id == "clamp_reversed_bounds", "train task differs")
    _require(held_out_trajectories[0].task_id == "merge_without_mutation", "held-out task differs")

    return {
        "schema_version": 1,
        "status": "v0_2_agent_sft_passed",
        "git_commit": collect_git_metadata(project_root)["commit"],
        "hardware": {
            "profile": hardware.id,
            "gpu": doctor["cuda"]["gpus"][0]["name"],
            "peak_reserved_gib": train_summary.peak_reserved_bytes / 1024**3,
            "hard_limit_gib": hardware.memory_budget.hard_peak_reserved_gib,
        },
        "dataset": {
            "sha256": dataset_manifest.dataset_sha256,
            "train_examples": dataset_manifest.train_examples,
            "validation_examples": dataset_manifest.validation_examples,
            "recovery_examples": dataset_manifest.recovery_examples,
            "exact_replays": dataset_manifest.exact_replays_passed,
            "max_sequence_tokens": max(len(item.input_ids) for item in examples),
        },
        "training": {
            "run_manifest_sha256": sha256_file(train_dir / "run_manifest.json"),
            "adapter_sha256": train_manifest["checkpoint"]["sha256"],
            "optimizer_steps": train_summary.optimizer_steps,
            "initial_validation_nll": train_summary.initial_validation_nll,
            "final_validation_nll": train_summary.final_validation_nll,
            "initial_validation_token_accuracy": train_summary.initial_validation_token_accuracy,
            "final_validation_token_accuracy": train_summary.final_validation_token_accuracy,
        },
        "behavior": {
            "base_train_action_validity": baseline.action_validity_rate,
            "adapted_train_action_validity": adapted.action_validity_rate,
            "adapted_train_score": adapted.mean_score,
            "snapshot_train_action_validity": snapshot.action_validity_rate,
            "snapshot_train_score": snapshot.mean_score,
            "held_out_action_validity": held_out.action_validity_rate,
            "held_out_score": held_out.mean_score,
            "held_out_solved": held_out.solved,
        },
        "checksums": {
            "files": len(checksums),
            "manifest_sha256": sha256_file(evidence_root / "checksums.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = validate_v0_2_agent_sft(args.evidence_root, args.project_root)
    if args.output:
        write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
