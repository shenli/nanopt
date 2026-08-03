"""Validate M7 pipeline lineage, reproducibility, memory, and protected evaluation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from nanopt.config.loader import ConfigRepository
from nanopt.data.preferences import read_preference_pairs
from nanopt.pipeline.records import PipelineManifest
from nanopt.runtime.artifacts import sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata
from nanopt.sft.checkpoint import sha256_directory
from scripts.validate_m3_reference_smoke import (
    _read_object,
    _require,
    _validate_dataset,
    _validate_doctor,
    _validate_run,
)

EXPECTED_STAGES = [
    "load_calibration",
    "eval_calibration",
    "base_eval",
    "sft_calibration",
    "sft",
    "sft_eval",
    "preferences",
    "dpo_calibration",
    "dpo",
    "dpo_eval",
    "grpo_calibration",
    "grpo",
    "grpo_eval",
    "grpo_eval_repeat",
    "report",
]


def _stage_map(manifest: PipelineManifest) -> dict[str, Any]:
    return {stage.id: stage for stage in manifest.stages}


def _verify_checksums(evidence_root: Path) -> dict[str, str]:
    checksums = _read_object(evidence_root / "checksums.json")
    _require(bool(checksums), "checksum manifest is empty")
    for relative, expected in checksums.items():
        _require(isinstance(relative, str), "checksum path is not a string")
        _require(isinstance(expected, str), f"checksum for {relative} is not a string")
        path = evidence_root / relative
        _require(path.is_file(), f"checksummed file is missing: {relative}")
        _require(sha256_file(path) == expected, f"checksum mismatch: {relative}")
    return {str(key): str(value) for key, value in checksums.items()}


def _validate_children(
    pipeline_dir: Path, manifest: PipelineManifest, project_root: Path
) -> dict[str, Path]:
    run_schema = _read_object(project_root / "specs/schemas/run_manifest.schema.json")
    children: dict[str, Path] = {}
    for stage in manifest.stages:
        _require(stage.status == "completed", f"stage did not complete: {stage.id}")
        _require(bool(stage.attempts), f"stage has no retained attempt: {stage.id}")
        for attempt in stage.attempts:
            _require(attempt.finished_at is not None, f"attempt has no finish time: {stage.id}")
            _require(attempt.wall_seconds >= 0, f"attempt has invalid timing: {stage.id}")
            if attempt.status == "failed":
                _require(attempt.failure is not None, f"failed attempt has no reason: {stage.id}")
        attempt = stage.attempts[-1]
        _require(attempt.status == "completed", f"last attempt failed: {stage.id}")
        if attempt.run_directory is None:
            continue
        run_dir = pipeline_dir / attempt.run_directory
        child = _read_object(run_dir / "run_manifest.json")
        jsonschema.Draft202012Validator(run_schema).validate(child)
        _require(child["status"] == "completed", f"child run failed: {stage.id}")
        _require(child["pipeline_run_id"] == manifest.pipeline_run_id, "parent link differs")
        _require(child["git"]["dirty"] is False, f"dirty child run: {stage.id}")
        _require(child["git"]["commit"] == manifest.git["commit"], "child commit differs")
        _require(
            sha256_file(run_dir / "run_manifest.json") == attempt.child_manifest_sha256,
            f"child manifest hash differs: {stage.id}",
        )
        children[stage.id] = run_dir
    return children


def _validate_stage_outputs(pipeline_dir: Path, manifest: PipelineManifest) -> None:
    for stage in manifest.stages:
        if stage.output_path is None:
            continue
        output = pipeline_dir / stage.output_path
        _require(output.exists(), f"stage output is missing: {stage.id}")
        actual = sha256_directory(output) if output.is_dir() else sha256_file(output)
        _require(actual == stage.output_checkpoint_sha256, f"stage output hash differs: {stage.id}")


def _evaluation_rates(run_dir: Path) -> dict[str, float | int]:
    summary = _read_object(run_dir / "summary.json")
    return {
        "examples": int(summary["examples"]),
        "accuracy": float(summary["accuracy"]["estimate"]),
        "parse_rate": float(summary["parse_rate"]["estimate"]),
    }


def validate_m7_reference_pipeline(evidence_root: Path, project_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    project_root = project_root.resolve()
    doctor = _validate_doctor(evidence_root, project_root)
    dataset_fingerprint, dataset_records = _validate_dataset(evidence_root, project_root)
    checksums = _verify_checksums(evidence_root)
    pipeline_dir = evidence_root / "pipelines/reference"
    raw_manifest = _read_object(pipeline_dir / "pipeline_manifest.json")
    schema = _read_object(project_root / "specs/schemas/pipeline_manifest.schema.json")
    jsonschema.Draft202012Validator(schema).validate(raw_manifest)
    manifest = PipelineManifest.model_validate(raw_manifest, strict=True)
    _require(manifest.status == "completed", "pipeline did not complete")
    _require(manifest.git["dirty"] is False, "pipeline checkout was dirty")
    current_git = collect_git_metadata(project_root)
    _require(manifest.git["commit"] == current_git["commit"], "pipeline commit differs")
    _require(manifest.dataset_fingerprint == dataset_fingerprint, "pipeline dataset differs")
    _require(manifest.protected_splits_used_for_training is False, "protected data was used")
    _require([stage.id for stage in manifest.stages] == EXPECTED_STAGES, "stage order differs")
    _require(manifest.total_wall_seconds is not None, "total wall time is missing")
    children = _validate_children(pipeline_dir, manifest, project_root)
    _validate_stage_outputs(pipeline_dir, manifest)
    stages = _stage_map(manifest)

    evaluation_expectations = {
        "eval_calibration": (False, 2, "base-calibration"),
        "base_eval": (True, 44, "base"),
        "sft_eval": (True, 44, "sft"),
        "dpo_eval": (True, 44, "dpo"),
        "grpo_eval": (True, 44, "grpo"),
        "grpo_eval_repeat": (True, 44, "grpo"),
    }
    for stage_id, (representative, count, checkpoint) in evaluation_expectations.items():
        _validate_run(
            children[stage_id],
            expected_representative=representative,
            expected_task_count=count,
            expected_checkpoint=checkpoint,
            expected_dataset_fingerprint=dataset_fingerprint,
            project_root=project_root,
        )
        evaluation = _read_object(children[stage_id] / "run_manifest.json")["evaluation"]
        _require(evaluation["device"] == "cuda", f"{stage_id} did not run on CUDA")

    for calibration in ("sft_calibration", "dpo_calibration", "grpo_calibration"):
        child = _read_object(children[calibration] / "run_manifest.json")
        _require(child["training"]["device"] == "cuda", f"{calibration} was not CUDA")
        _require(child["training"]["representative"] is False, f"{calibration} mislabeled")
    for training in ("sft", "dpo", "grpo"):
        child = _read_object(children[training] / "run_manifest.json")
        _require(child["training"]["device"] == "cuda", f"{training} was not CUDA")
        _require(child["training"]["representative"] is True, f"{training} mislabeled")

    load = _read_object(children["load_calibration"] / "summary.json")
    _require(load["device"] == "cuda", "load calibration did not run on CUDA")
    _require(load["representative"] is False, "load calibration mislabeled")
    sft_sha = str(stages["sft"].output_checkpoint_sha256)
    dpo_sha = str(stages["dpo"].output_checkpoint_sha256)
    grpo_sha = str(stages["grpo"].output_checkpoint_sha256)
    _require(stages["sft_eval"].input_checkpoint_sha256 == sft_sha, "SFT lineage differs")
    _require(stages["dpo"].input_checkpoint_sha256 == sft_sha, "DPO parent differs")
    _require(stages["dpo_eval"].input_checkpoint_sha256 == dpo_sha, "DPO eval differs")
    _require(stages["grpo"].input_checkpoint_sha256 == dpo_sha, "GRPO parent differs")
    _require(stages["grpo_eval"].input_checkpoint_sha256 == grpo_sha, "GRPO eval differs")
    _require(manifest.final_checkpoint_sha256 == grpo_sha, "final checkpoint differs")
    for stage_id, expected_sha in (
        ("sft_eval", sft_sha),
        ("dpo_eval", dpo_sha),
        ("grpo_eval", grpo_sha),
    ):
        child = _read_object(children[stage_id] / "run_manifest.json")
        _require(child["model"]["adapter_sha256"] == expected_sha, f"{stage_id} adapter differs")

    pairs = read_preference_pairs(pipeline_dir / "data/preferences.jsonl")
    _require(all(pair.split in {"train", "validation"} for pair in pairs), "protected preference")
    comparison = _read_object(pipeline_dir / "comparison.json")
    _require(
        comparison["final_evaluation_repeat"]["exact_generation_match"] is True,
        "repeated final generation differs",
    )
    rates = {
        name: _evaluation_rates(children[stage_id])
        for name, stage_id in {
            "base": "base_eval",
            "sft": "sft_eval",
            "dpo": "dpo_eval",
            "grpo": "grpo_eval",
        }.items()
    }
    targets_value = yaml.safe_load((project_root / "configs/reference_targets.yaml").read_text())
    m6_targets = targets_value["m6"]
    _require(
        float(rates["grpo"]["accuracy"])
        >= float(rates["dpo"]["accuracy"])
        - float(m6_targets["maximum_overall_accuracy_regression"]),
        "GRPO regression exceeds frozen target",
    )
    hard_budget = (
        ConfigRepository(project_root / "configs")
        .hardware("rtx_4070_ti_super_16gb")
        .memory_budget.hard_peak_reserved_gib
        * 1024**3
    )
    _require(bool(manifest.phase_peak_reserved_bytes), "phase memory measurements missing")
    _require(
        max(manifest.phase_peak_reserved_bytes.values()) < hard_budget,
        "pipeline exceeded hard memory budget",
    )
    failed_attempts = sum(
        attempt.status == "failed" for stage in manifest.stages for attempt in stage.attempts
    )
    _require(
        failed_attempts == len(manifest.failures_and_retries),
        "failure/retry disclosure count differs",
    )
    for name, expected in manifest.comparison_artifacts.items():
        _require(sha256_file(pipeline_dir / name) == expected, f"report hash differs: {name}")

    return {
        "schema_version": 1,
        "status": "m7_reference_pipeline_passed",
        "git_commit": manifest.git["commit"],
        "hardware": {
            "profile": manifest.hardware_id,
            "gpu": doctor["cuda"]["gpus"][0]["name"],
            "driver_version": doctor["cuda"]["driver_version"],
            "runtime_version": doctor["cuda"]["runtime_version"],
        },
        "dataset": {"records": dataset_records, "fingerprint": dataset_fingerprint},
        "pipeline": {
            "run_id": manifest.pipeline_run_id,
            "manifest_sha256": sha256_file(pipeline_dir / "pipeline_manifest.json"),
            "stages": len(manifest.stages),
            "failed_attempts": failed_attempts,
            "total_wall_seconds": manifest.total_wall_seconds,
            "phase_peak_reserved_bytes": manifest.phase_peak_reserved_bytes,
            "final_checkpoint_sha256": manifest.final_checkpoint_sha256,
        },
        "evaluation": rates,
        "repeat": comparison["final_evaluation_repeat"],
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
    evidence = validate_m7_reference_pipeline(args.evidence_root, args.project_root)
    if args.output:
        write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
