"""Validate M4 training, adapter lineage, memory, and protected generation evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jsonschema

from nanopt.config.loader import ConfigRepository
from nanopt.runtime.artifacts import read_jsonl, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata
from nanopt.sft.checkpoint import sha256_directory
from nanopt.sft.records import SftMetricRecord, SftSummary
from scripts.validate_m3_reference_smoke import (
    _read_object,
    _require,
    _validate_dataset,
    _validate_doctor,
    _validate_run,
)

MINIMUM_PARSE_RATE = 0.50
MINIMUM_ACCURACY = 0.05


def _validate_training_run(
    run_dir: Path,
    *,
    project_root: Path,
    expected_train_examples: int,
    expected_validation_examples: int,
    expected_representative: bool,
    expected_dataset_fingerprint: str,
) -> dict[str, Any]:
    manifest = _read_object(run_dir / "run_manifest.json")
    manifest_schema = _read_object(project_root / "specs/schemas/run_manifest.schema.json")
    jsonschema.Draft202012Validator(manifest_schema).validate(manifest)
    _require(manifest.get("status") == "completed", f"SFT run did not complete: {run_dir.name}")
    _require(manifest.get("stage") == "sft", "training run has the wrong stage")
    git = manifest.get("git")
    current_git = collect_git_metadata(project_root)
    _require(isinstance(git, dict) and git.get("dirty") is False, "SFT checkout was dirty")
    _require(git.get("commit") == current_git["commit"], "SFT run commit differs from checkout")

    training = manifest.get("training")
    _require(isinstance(training, dict), "SFT manifest is missing training evidence")
    _require(training.get("device") == "cuda", "SFT did not execute on CUDA")
    _require(
        training.get("train_examples") == expected_train_examples,
        "SFT training example count is wrong",
    )
    _require(
        training.get("validation_examples") == expected_validation_examples,
        "SFT validation example count is wrong",
    )
    _require(
        training.get("representative") is expected_representative,
        "SFT representative label is wrong",
    )
    _require(
        manifest.get("data", {}).get("fingerprints", {}).get("dataset")
        == expected_dataset_fingerprint,
        "SFT dataset fingerprint is wrong",
    )

    model = manifest.get("model")
    _require(isinstance(model, dict) and model.get("adapter_name") == "sft", "SFT adapter missing")
    adapter_sha = model.get("adapter_sha256")
    _require(isinstance(adapter_sha, str) and len(adapter_sha) == 64, "SFT adapter hash missing")
    checkpoint = manifest.get("checkpoint")
    _require(isinstance(checkpoint, dict), "SFT manifest is missing final checkpoint")
    adapter_path = run_dir / str(checkpoint.get("path"))
    _require(sha256_directory(adapter_path) == adapter_sha, "SFT adapter files do not match hash")

    metric_schema = _read_object(project_root / "specs/schemas/sft_metric.schema.json")
    raw_metrics = read_jsonl(run_dir / "metrics.jsonl")
    metrics: list[SftMetricRecord] = []
    for raw in raw_metrics:
        jsonschema.Draft202012Validator(metric_schema).validate(raw)
        metrics.append(SftMetricRecord.model_validate(raw, strict=True))
    _require(any(metric.split == "train" for metric in metrics), "SFT run has no train metrics")
    _require(
        sum(metric.split == "validation" for metric in metrics) >= 2,
        "SFT run needs initial and final validation metrics",
    )

    summary = SftSummary.model_validate(_read_object(run_dir / "summary.json"), strict=True)
    _require(summary.representative is expected_representative, "SFT summary label is wrong")
    _require(
        summary.final_validation_nll < summary.initial_validation_nll,
        "SFT did not lower held-out completion NLL",
    )
    hard_budget = (
        ConfigRepository(project_root / "configs")
        .hardware("rtx_4070_ti_super_16gb")
        .memory_budget.hard_peak_reserved_gib
    )
    _require(
        summary.peak_reserved_bytes < hard_budget * 1024**3,
        "SFT exceeded the hard reserved-memory budget",
    )
    return {
        "run_id": manifest["run_id"],
        "representative": expected_representative,
        "optimizer_steps": summary.optimizer_steps,
        "initial_validation_nll": summary.initial_validation_nll,
        "final_validation_nll": summary.final_validation_nll,
        "peak_reserved_bytes": summary.peak_reserved_bytes,
        "adapter_sha256": adapter_sha,
    }


def validate_m4_reference_sft(evidence_root: Path, project_root: Path) -> dict[str, Any]:
    """Validate the complete M4 bundle and return its compact public summary."""

    evidence_root = evidence_root.resolve()
    project_root = project_root.resolve()
    doctor = _validate_doctor(evidence_root, project_root)
    dataset_fingerprint, dataset_records = _validate_dataset(evidence_root, project_root)
    calibration = _validate_training_run(
        evidence_root / "runs/calibration-sft",
        project_root=project_root,
        expected_train_examples=2,
        expected_validation_examples=16,
        expected_representative=False,
        expected_dataset_fingerprint=dataset_fingerprint,
    )
    training = _validate_training_run(
        evidence_root / "runs/sft",
        project_root=project_root,
        expected_train_examples=64,
        expected_validation_examples=16,
        expected_representative=True,
        expected_dataset_fingerprint=dataset_fingerprint,
    )
    evaluation = _validate_run(
        evidence_root / "runs/sft-eval",
        expected_representative=True,
        expected_task_count=44,
        expected_checkpoint="sft",
        expected_dataset_fingerprint=dataset_fingerprint,
        project_root=project_root,
    )
    eval_manifest = _read_object(evidence_root / "runs/sft-eval/run_manifest.json")
    _require(
        eval_manifest.get("model", {}).get("adapter_sha256") == training["adapter_sha256"],
        "evaluation did not load the trained adapter",
    )
    eval_summary = _read_object(evidence_root / "runs/sft-eval/summary.json")
    parse_rate = float(eval_summary.get("parse_rate", {}).get("estimate", 0.0))
    accuracy = float(eval_summary.get("accuracy", {}).get("estimate", 0.0))
    protocol_stop_rate = float(eval_summary.get("stop_sequence_fraction", 0.0))
    _require(parse_rate >= MINIMUM_PARSE_RATE, "SFT protected parse rate missed the M4 target")
    _require(accuracy >= MINIMUM_ACCURACY, "SFT protected accuracy missed the M4 target")
    _require(
        protocol_stop_rate >= MINIMUM_PARSE_RATE,
        "SFT protocol stop rate missed the M4 target",
    )

    return {
        "schema_version": 1,
        "status": "m4_reference_sft_passed",
        "git_commit": collect_git_metadata(project_root)["commit"],
        "hardware": {
            "profile": "rtx_4070_ti_super_16gb",
            "gpu": doctor["cuda"]["gpus"][0]["name"],
            "driver_version": doctor["cuda"]["driver_version"],
            "runtime_version": doctor["cuda"]["runtime_version"],
        },
        "dataset": {"records": dataset_records, "fingerprint": dataset_fingerprint},
        "targets": {"minimum_parse_rate": MINIMUM_PARSE_RATE, "minimum_accuracy": MINIMUM_ACCURACY},
        "calibration": calibration,
        "training": training,
        "evaluation": {
            **evaluation,
            "parse_rate": parse_rate,
            "accuracy": accuracy,
            "protocol_stop_rate": protocol_stop_rate,
            "summary_sha256": sha256_file(evidence_root / "runs/sft-eval/summary.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    evidence = validate_m4_reference_sft(arguments.evidence_root, project_root)
    write_json(arguments.output, evidence)
    print(f"M4 reference SFT validation passed; wrote {arguments.output}")


if __name__ == "__main__":
    main()
