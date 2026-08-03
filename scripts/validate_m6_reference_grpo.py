"""Validate exact-token M6 trajectories, GRPO numerics, rewards, lineage, and transfer."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from nanopt.config.loader import ConfigRepository
from nanopt.eval.io import read_arithmetic_tasks
from nanopt.grpo.records import GrpoMetricRecord, GrpoSummary, GrpoTrajectoryRecord
from nanopt.runtime.artifacts import read_jsonl, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata
from nanopt.sft.checkpoint import sha256_directory
from scripts.validate_m3_reference_smoke import (
    _read_object,
    _require,
    _validate_dataset,
    _validate_doctor,
    _validate_run,
)


def _targets(project_root: Path) -> dict[str, Any]:
    value = yaml.safe_load((project_root / "configs/reference_targets.yaml").read_text())
    _require(isinstance(value, dict) and isinstance(value.get("m6"), dict), "M6 targets missing")
    return value["m6"]


def _validate_reward_hacking(path: Path, maximum_credit: float) -> dict[str, Any]:
    values = _read_object(path)
    _require(isinstance(values, list) and values, "reward-hacking suite is empty")
    for value in values:
        _require(isinstance(value, dict), "reward-hacking result is malformed")
        _require(value.get("passed") is True, "reward-hacking case did not pass")
        _require(
            float(value.get("correctness_reward", 1.0)) <= maximum_credit,
            "reward-hacking case received correctness credit",
        )
    return {"cases": len(values), "sha256": sha256_file(path)}


def _validate_grpo_run(
    run_dir: Path,
    *,
    project_root: Path,
    expected_iterations: int,
    expected_group_size: int,
    expected_representative: bool,
    expected_dataset_fingerprint: str,
    targets: dict[str, Any],
) -> dict[str, Any]:
    manifest = _read_object(run_dir / "run_manifest.json")
    manifest_schema = _read_object(project_root / "specs/schemas/run_manifest.schema.json")
    jsonschema.Draft202012Validator(manifest_schema).validate(manifest)
    _require(manifest.get("status") == "completed", f"GRPO run did not complete: {run_dir.name}")
    _require(manifest.get("stage") == "grpo", "GRPO run has the wrong stage")
    current_git = collect_git_metadata(project_root)
    git = manifest.get("git", {})
    _require(git.get("dirty") is False, "GRPO checkout was dirty")
    _require(git.get("commit") == current_git["commit"], "GRPO commit differs from checkout")
    training = manifest.get("training", {})
    expected_trajectories = expected_iterations
    expected_completions = expected_trajectories * expected_group_size
    _require(training.get("device") == "cuda", "GRPO did not execute on CUDA")
    _require(training.get("iterations") == expected_iterations, "GRPO iteration count is wrong")
    _require(training.get("trajectories") == expected_trajectories, "trajectory count is wrong")
    _require(training.get("completions") == expected_completions, "completion count is wrong")
    _require(training.get("representative") is expected_representative, "GRPO label is wrong")
    _require(training.get("consumed_exact_stored_token_ids") is True, "stored IDs not consumed")
    _require(
        manifest.get("data", {}).get("fingerprints", {}).get("dataset")
        == expected_dataset_fingerprint,
        "GRPO dataset fingerprint changed",
    )

    model = manifest.get("model", {})
    parent_sha = model.get("parent_adapter_sha256")
    adapter_sha = model.get("adapter_sha256")
    _require(isinstance(parent_sha, str) and len(parent_sha) == 64, "DPO parent hash missing")
    _require(isinstance(adapter_sha, str) and len(adapter_sha) == 64, "GRPO adapter hash missing")
    _require(sha256_directory(run_dir / "adapter/grpo") == adapter_sha, "GRPO adapter hash differs")

    trajectory_schema = _read_object(project_root / "specs/schemas/rlvr_trajectory.schema.json")
    task_splits = {
        task.task_id: task.split
        for task in read_arithmetic_tasks(run_dir.parents[1] / "data/tasks.jsonl")
    }
    trajectories: list[GrpoTrajectoryRecord] = []
    for raw in read_jsonl(run_dir / "trajectories.jsonl"):
        jsonschema.Draft202012Validator(trajectory_schema).validate(raw)
        trajectory = GrpoTrajectoryRecord.model_validate(raw, strict=True)
        _require(task_splits.get(trajectory.task_id) == "train", "protected task entered rollout")
        _require(len(trajectory.completions) == expected_group_size, "group size changed")
        for completion in trajectory.completions:
            _require(
                len(completion.token_ids)
                == len(completion.action_mask)
                == len(completion.old_logprobs),
                "trajectory token coordinates differ",
            )
            _require(all(completion.action_mask), "stored completion contains inactive action")
            _require(
                all(math.isfinite(value) for value in completion.old_logprobs), "old logp nonfinite"
            )
            _require(math.isfinite(completion.reward), "reward nonfinite")
            _require(math.isfinite(completion.advantage), "advantage nonfinite")
        trajectories.append(trajectory)
    _require(len(trajectories) == expected_trajectories, "trajectory JSONL is incomplete")

    metric_schema = _read_object(project_root / "specs/schemas/grpo_metric.schema.json")
    metrics: list[GrpoMetricRecord] = []
    for raw in read_jsonl(run_dir / "metrics.jsonl"):
        jsonschema.Draft202012Validator(metric_schema).validate(raw)
        metric = GrpoMetricRecord.model_validate(raw, strict=True)
        _require(
            all(
                math.isfinite(float(value))
                for value in metric.model_dump().values()
                if isinstance(value, (int, float))
            ),
            "metric nonfinite",
        )
        metrics.append(metric)
    _require(len(metrics) == expected_iterations, "GRPO metric iteration count differs")
    summary = GrpoSummary.model_validate(_read_object(run_dir / "summary.json"), strict=True)
    _require(summary.representative is expected_representative, "GRPO summary label is wrong")
    _require(summary.advantage_mode == "group_zscore", "advantage mode changed")
    _require(summary.loss_normalization == "token_mean", "loss normalization changed")
    _require(summary.clip_epsilon == 0.2, "clip epsilon changed")
    _require(summary.kl_beta == 0, "reference KL beta changed")
    hard_budget = (
        ConfigRepository(project_root / "configs")
        .hardware("rtx_4070_ti_super_16gb")
        .memory_budget.hard_peak_reserved_gib
    )
    _require(summary.peak_reserved_bytes < hard_budget * 1024**3, "GRPO exceeded memory budget")
    if expected_representative:
        _require(
            summary.parser_success_rate >= float(targets["minimum_rollout_parser_success_rate"]),
            "rollout parser success missed target",
        )
        _require(
            summary.correctness_rate >= float(targets["minimum_rollout_correctness_rate"]),
            "rollout correctness missed target",
        )
        _require(
            summary.degenerate_group_fraction
            <= float(targets["maximum_degenerate_group_fraction"]),
            "degenerate group fraction exceeded target",
        )
        _require(
            summary.mean_clip_fraction <= float(targets["maximum_mean_clip_fraction"]),
            "clip fraction exceeded target",
        )
        _require(
            max(metric.ratio_p95 for metric in metrics) <= float(targets["maximum_ratio_p95"]),
            "ratio p95 exceeded target",
        )
    reward_hacking = _validate_reward_hacking(
        run_dir / "reward_hacking.json",
        float(targets["maximum_reward_hacking_correctness_credit"]),
    )
    return {
        "run_id": manifest["run_id"],
        "representative": summary.representative,
        "iterations": summary.iterations,
        "optimizer_steps": summary.optimizer_steps,
        "trajectories": summary.trajectories,
        "completions": summary.completions,
        "mean_reward": summary.mean_reward,
        "correctness_rate": summary.correctness_rate,
        "parser_success_rate": summary.parser_success_rate,
        "degenerate_group_fraction": summary.degenerate_group_fraction,
        "mean_clip_fraction": summary.mean_clip_fraction,
        "maximum_ratio_p95": max(metric.ratio_p95 for metric in metrics),
        "peak_reserved_bytes": summary.peak_reserved_bytes,
        "parent_dpo_adapter_sha256": parent_sha,
        "grpo_adapter_sha256": adapter_sha,
        "trajectories_sha256": sha256_file(run_dir / "trajectories.jsonl"),
        "reward_hacking": reward_hacking,
    }


def _split_accuracy(run_dir: Path) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in read_jsonl(run_dir / "samples.jsonl"):
        groups[str(result["split"])].append(result)
    return {
        split: {
            "correct": sum(item["verifier_status"] == "correct" for item in values),
            "count": len(values),
            "accuracy": sum(item["verifier_status"] == "correct" for item in values) / len(values),
        }
        for split, values in sorted(groups.items())
    }


def _summary_rates(run_dir: Path) -> tuple[float, float]:
    summary = _read_object(run_dir / "summary.json")
    return (
        float(summary["accuracy"]["estimate"]),
        float(summary["parse_rate"]["estimate"]),
    )


def validate_m6_reference_grpo(evidence_root: Path, project_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    project_root = project_root.resolve()
    targets = _targets(project_root)
    doctor = _validate_doctor(evidence_root, project_root)
    dataset_fingerprint, dataset_records = _validate_dataset(evidence_root, project_root)
    calibration = _validate_grpo_run(
        evidence_root / "runs/calibration-grpo",
        project_root=project_root,
        expected_iterations=1,
        expected_group_size=2,
        expected_representative=False,
        expected_dataset_fingerprint=dataset_fingerprint,
        targets=targets,
    )
    training = _validate_grpo_run(
        evidence_root / "runs/grpo",
        project_root=project_root,
        expected_iterations=int(targets["iterations"]),
        expected_group_size=int(targets["group_size"]),
        expected_representative=True,
        expected_dataset_fingerprint=dataset_fingerprint,
        targets=targets,
    )
    _require(
        calibration["parent_dpo_adapter_sha256"] == training["parent_dpo_adapter_sha256"],
        "calibration parent differs",
    )
    dpo_eval = _validate_run(
        evidence_root / "runs/dpo-eval",
        expected_representative=True,
        expected_task_count=44,
        expected_checkpoint="dpo",
        expected_dataset_fingerprint=dataset_fingerprint,
        project_root=project_root,
    )
    grpo_eval = _validate_run(
        evidence_root / "runs/grpo-eval",
        expected_representative=True,
        expected_task_count=44,
        expected_checkpoint="grpo",
        expected_dataset_fingerprint=dataset_fingerprint,
        project_root=project_root,
    )
    dpo_manifest = _read_object(evidence_root / "runs/dpo-eval/run_manifest.json")
    grpo_manifest = _read_object(evidence_root / "runs/grpo-eval/run_manifest.json")
    _require(
        dpo_manifest["model"]["adapter_sha256"] == training["parent_dpo_adapter_sha256"],
        "DPO eval parent differs",
    )
    _require(
        grpo_manifest["model"]["adapter_sha256"] == training["grpo_adapter_sha256"],
        "GRPO eval adapter differs",
    )
    dpo_accuracy, dpo_parse = _summary_rates(evidence_root / "runs/dpo-eval")
    grpo_accuracy, grpo_parse = _summary_rates(evidence_root / "runs/grpo-eval")
    _require(
        grpo_accuracy >= dpo_accuracy - float(targets["maximum_overall_accuracy_regression"]),
        "overall GRPO accuracy regression exceeded target",
    )
    _require(
        grpo_parse >= dpo_parse - float(targets["maximum_overall_parse_regression"]),
        "overall GRPO parse regression exceeded target",
    )
    dpo_splits = _split_accuracy(evidence_root / "runs/dpo-eval")
    grpo_splits = _split_accuracy(evidence_root / "runs/grpo-eval")
    improved = [
        split
        for split in targets["primary_splits"]
        if float(grpo_splits[split]["accuracy"]) > float(dpo_splits[split]["accuracy"])
    ]
    _require(
        len(improved) >= int(targets["minimum_improved_primary_splits"]),
        "GRPO did not improve a frozen primary target split",
    )
    return {
        "schema_version": 1,
        "status": "m6_reference_grpo_passed",
        "git_commit": collect_git_metadata(project_root)["commit"],
        "hardware": {
            "profile": "rtx_4070_ti_super_16gb",
            "gpu": doctor["cuda"]["gpus"][0]["name"],
            "driver_version": doctor["cuda"]["driver_version"],
            "runtime_version": doctor["cuda"]["runtime_version"],
        },
        "dataset": {"records": dataset_records, "fingerprint": dataset_fingerprint},
        "targets": targets,
        "calibration": calibration,
        "training": training,
        "evaluation": {
            "dpo": {
                **dpo_eval,
                "accuracy": dpo_accuracy,
                "parse_rate": dpo_parse,
                "splits": dpo_splits,
            },
            "grpo": {
                **grpo_eval,
                "accuracy": grpo_accuracy,
                "parse_rate": grpo_parse,
                "splits": grpo_splits,
            },
            "improved_primary_splits": improved,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    evidence = validate_m6_reference_grpo(arguments.evidence_root, project_root)
    write_json(arguments.output, evidence)
    print(f"M6 reference GRPO validation passed; wrote {arguments.output}")


if __name__ == "__main__":
    main()
