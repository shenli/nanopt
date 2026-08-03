"""Validate M5 preference, cache, DPO, memory, lineage, and regression evidence."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema

from nanopt.config.loader import ConfigRepository
from nanopt.data.preferences import PreferenceAudit, PreferencePair
from nanopt.dpo.cache import ReferenceCacheManifest
from nanopt.dpo.records import DpoMetricRecord, DpoSummary
from nanopt.eval.io import read_arithmetic_tasks
from nanopt.eval.verifier import verify_task_response
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

MAXIMUM_PROTECTED_REGRESSION = 0.10
MAXIMUM_CACHE_PARITY_ERROR = 1e-5
MINIMUM_LENGTH_RATIO = 0.75
MAXIMUM_LENGTH_RATIO = 1.35


def _validate_preferences(
    evidence_root: Path, project_root: Path, source_fingerprint: str
) -> dict[str, Any]:
    data_dir = evidence_root / "data"
    raw_pairs = read_jsonl(data_dir / "preferences.jsonl")
    schema = _read_object(project_root / "specs/schemas/preference_pair.schema.json")
    pairs: list[PreferencePair] = []
    for raw in raw_pairs:
        jsonschema.Draft202012Validator(schema).validate(raw)
        pairs.append(PreferencePair.model_validate(raw, strict=True))
    audit = PreferenceAudit.model_validate(
        _read_object(data_dir / "preference_audit.json"), strict=True
    )
    _require(len(pairs) == audit.pair_count == 80, "preference dataset must contain 80 pairs")
    _require(audit.source_dataset_fingerprint == source_fingerprint, "preference source changed")
    _require({pair.split for pair in pairs} == {"train", "validation"}, "protected pair leaked")
    _require(len({pair.pair_id for pair in pairs}) == len(pairs), "duplicate preference pair ID")
    tasks = {task.task_id: task for task in read_arithmetic_tasks(data_dir / "tasks.jsonl")}
    for pair in pairs:
        task = tasks[pair.task_id]
        _require(verify_task_response(task, pair.chosen).correct, "chosen preference is incorrect")
        rejected = verify_task_response(task, pair.rejected)
        if pair.rejection_type == "wrong_answer":
            expected_failure = rejected.parser.valid and not rejected.correct
        elif pair.rejection_type == "malformed_answer":
            expected_failure = rejected.parser.status == "malformed_answer"
        else:
            expected_failure = rejected.parser.status == "trailing_content"
        _require(expected_failure, f"rejected failure contract changed for {pair.pair_id}")
    counts = Counter(pair.rejection_type for pair in pairs)
    _require(
        set(counts) == {"wrong_answer", "malformed_answer", "trailing_content"},
        "missing rejection type",
    )
    _require(max(counts.values()) - min(counts.values()) <= 1, "rejection mixture is imbalanced")
    return {
        "pairs": len(pairs),
        "fingerprint": audit.dataset_fingerprint,
        "rejection_type_counts": dict(sorted(counts.items())),
        "character_length_ratio": audit.rejected_to_chosen_character_ratio,
        "preferences_sha256": sha256_file(data_dir / "preferences.jsonl"),
        "audit_sha256": sha256_file(data_dir / "preference_audit.json"),
    }


def _validate_dpo_run(
    run_dir: Path,
    *,
    project_root: Path,
    expected_train_pairs: int,
    expected_validation_pairs: int,
    expected_cache_entries: int,
    expected_representative: bool,
    expected_preference_fingerprint: str,
) -> dict[str, Any]:
    manifest = _read_object(run_dir / "run_manifest.json")
    schema = _read_object(project_root / "specs/schemas/run_manifest.schema.json")
    jsonschema.Draft202012Validator(schema).validate(manifest)
    _require(manifest.get("status") == "completed", f"DPO run did not complete: {run_dir.name}")
    _require(manifest.get("stage") == "dpo", "DPO run has the wrong stage")
    git = manifest.get("git", {})
    current_git = collect_git_metadata(project_root)
    _require(git.get("dirty") is False, "DPO checkout was dirty")
    _require(git.get("commit") == current_git["commit"], "DPO run commit differs from checkout")
    training = manifest.get("training", {})
    _require(training.get("device") == "cuda", "DPO did not execute on CUDA")
    _require(training.get("train_pairs") == expected_train_pairs, "DPO train pair count is wrong")
    _require(
        training.get("validation_pairs") == expected_validation_pairs,
        "DPO validation pair count is wrong",
    )
    _require(training.get("representative") is expected_representative, "DPO run label is wrong")
    _require(training.get("initial_policy_is_exact_sft_copy") is True, "DPO did not start from SFT")
    fingerprints = manifest.get("data", {}).get("fingerprints", {})
    _require(
        fingerprints.get("preference_dataset") == expected_preference_fingerprint,
        "DPO preference fingerprint changed",
    )
    model = manifest.get("model", {})
    sft_sha = model.get("parent_adapter_sha256")
    dpo_sha = model.get("adapter_sha256")
    _require(isinstance(sft_sha, str) and len(sft_sha) == 64, "SFT parent hash is missing")
    _require(isinstance(dpo_sha, str) and len(dpo_sha) == 64, "DPO adapter hash is missing")
    _require(sha256_directory(run_dir / "adapter/dpo") == dpo_sha, "DPO adapter hash differs")

    cache_manifest = ReferenceCacheManifest.model_validate(
        _read_object(run_dir / "reference_cache/cache_manifest.json"), strict=True
    )
    _require(cache_manifest.entry_count == expected_cache_entries, "reference cache is incomplete")
    _require(
        cache_manifest.identity.sft_adapter_sha256 == sft_sha, "cache used another SFT adapter"
    )
    _require(
        cache_manifest.identity.preference_dataset_fingerprint == expected_preference_fingerprint,
        "cache used another preference dataset",
    )
    entries = read_jsonl(run_dir / "reference_cache/reference_logps.jsonl")
    _require(len(entries) == expected_cache_entries, "reference cache JSONL count differs")
    chosen_mean = sum(int(entry["chosen_active_tokens"]) for entry in entries) / len(entries)
    rejected_mean = sum(int(entry["rejected_active_tokens"]) for entry in entries) / len(entries)
    token_length_ratio = rejected_mean / chosen_mean
    _require(
        MINIMUM_LENGTH_RATIO <= token_length_ratio <= MAXIMUM_LENGTH_RATIO,
        "chosen/rejected token lengths exceed the frozen audit tolerance",
    )

    metric_schema = _read_object(project_root / "specs/schemas/dpo_metric.schema.json")
    metrics = []
    for raw in read_jsonl(run_dir / "metrics.jsonl"):
        jsonschema.Draft202012Validator(metric_schema).validate(raw)
        metrics.append(DpoMetricRecord.model_validate(raw, strict=True))
    _require(any(metric.split == "train" for metric in metrics), "DPO run has no train metrics")
    _require(sum(metric.split == "validation" for metric in metrics) >= 2, "DPO validation missing")
    summary = DpoSummary.model_validate(_read_object(run_dir / "summary.json"), strict=True)
    _require(summary.representative is expected_representative, "DPO summary label is wrong")
    _require(
        abs(summary.initial_validation_loss - math.log(2)) <= 1e-5,
        "initial DPO loss does not prove an exact SFT copy",
    )
    _require(
        summary.reference_cache_parity_max_abs_error <= MAXIMUM_CACHE_PARITY_ERROR,
        "reference cache/live parity failed",
    )
    _require(
        summary.final_validation_loss < summary.initial_validation_loss,
        "held-out DPO loss did not improve",
    )
    _require(
        summary.final_validation_policy_margin > summary.initial_validation_policy_margin,
        "held-out chosen margin did not improve",
    )
    hard_budget = (
        ConfigRepository(project_root / "configs")
        .hardware("rtx_4070_ti_super_16gb")
        .memory_budget.hard_peak_reserved_gib
    )
    _require(summary.peak_reserved_bytes < hard_budget * 1024**3, "DPO exceeded memory budget")
    breakdown = _read_object(run_dir / "preference_breakdown.json")
    _require(
        set(breakdown) == {"wrong_answer", "malformed_answer", "trailing_content"},
        "DPO rejection-type breakdown is incomplete",
    )
    _require(
        all(int(value.get("pair_count", 0)) > 0 for value in breakdown.values()),
        "DPO rejection-type breakdown contains an empty group",
    )
    return {
        "run_id": manifest["run_id"],
        "representative": summary.representative,
        "optimizer_steps": summary.optimizer_steps,
        "initial_validation_loss": summary.initial_validation_loss,
        "final_validation_loss": summary.final_validation_loss,
        "initial_validation_margin": summary.initial_validation_policy_margin,
        "final_validation_margin": summary.final_validation_policy_margin,
        "final_reward_accuracy": summary.final_validation_reward_accuracy,
        "cache_entries": cache_manifest.entry_count,
        "cache_sha256": cache_manifest.cache_sha256,
        "cache_parity_max_abs_error": summary.reference_cache_parity_max_abs_error,
        "token_length_ratio": token_length_ratio,
        "peak_reserved_bytes": summary.peak_reserved_bytes,
        "sft_adapter_sha256": sft_sha,
        "dpo_adapter_sha256": dpo_sha,
    }


def _evaluation_rates(run_dir: Path) -> tuple[float, float]:
    summary = _read_object(run_dir / "summary.json")
    return (
        float(summary.get("parse_rate", {}).get("estimate", 0.0)),
        float(summary.get("accuracy", {}).get("estimate", 0.0)),
    )


def validate_m5_reference_dpo(evidence_root: Path, project_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    project_root = project_root.resolve()
    doctor = _validate_doctor(evidence_root, project_root)
    dataset_fingerprint, dataset_records = _validate_dataset(evidence_root, project_root)
    preferences = _validate_preferences(evidence_root, project_root, dataset_fingerprint)
    calibration = _validate_dpo_run(
        evidence_root / "runs/calibration-dpo",
        project_root=project_root,
        expected_train_pairs=2,
        expected_validation_pairs=16,
        expected_cache_entries=18,
        expected_representative=False,
        expected_preference_fingerprint=preferences["fingerprint"],
    )
    training = _validate_dpo_run(
        evidence_root / "runs/dpo",
        project_root=project_root,
        expected_train_pairs=64,
        expected_validation_pairs=16,
        expected_cache_entries=80,
        expected_representative=True,
        expected_preference_fingerprint=preferences["fingerprint"],
    )
    _require(
        calibration["sft_adapter_sha256"] == training["sft_adapter_sha256"],
        "calibration parent differs",
    )
    sft_eval = _validate_run(
        evidence_root / "runs/sft-eval",
        expected_representative=True,
        expected_task_count=44,
        expected_checkpoint="sft",
        expected_dataset_fingerprint=dataset_fingerprint,
        project_root=project_root,
    )
    dpo_eval = _validate_run(
        evidence_root / "runs/dpo-eval",
        expected_representative=True,
        expected_task_count=44,
        expected_checkpoint="dpo",
        expected_dataset_fingerprint=dataset_fingerprint,
        project_root=project_root,
    )
    sft_parse, sft_accuracy = _evaluation_rates(evidence_root / "runs/sft-eval")
    dpo_parse, dpo_accuracy = _evaluation_rates(evidence_root / "runs/dpo-eval")
    _require(
        dpo_parse >= sft_parse - MAXIMUM_PROTECTED_REGRESSION, "DPO parse regression exceeds target"
    )
    _require(
        dpo_accuracy >= sft_accuracy - MAXIMUM_PROTECTED_REGRESSION,
        "DPO accuracy regression exceeds target",
    )
    return {
        "schema_version": 1,
        "status": "m5_reference_dpo_passed",
        "git_commit": collect_git_metadata(project_root)["commit"],
        "hardware": {
            "profile": "rtx_4070_ti_super_16gb",
            "gpu": doctor["cuda"]["gpus"][0]["name"],
            "driver_version": doctor["cuda"]["driver_version"],
            "runtime_version": doctor["cuda"]["runtime_version"],
        },
        "dataset": {"records": dataset_records, "fingerprint": dataset_fingerprint},
        "preferences": preferences,
        "targets": {
            "maximum_cache_parity_error": MAXIMUM_CACHE_PARITY_ERROR,
            "maximum_protected_regression": MAXIMUM_PROTECTED_REGRESSION,
            "token_length_ratio": [MINIMUM_LENGTH_RATIO, MAXIMUM_LENGTH_RATIO],
        },
        "calibration": calibration,
        "training": training,
        "evaluation": {
            "sft": {**sft_eval, "parse_rate": sft_parse, "accuracy": sft_accuracy},
            "dpo": {**dpo_eval, "parse_rate": dpo_parse, "accuracy": dpo_accuracy},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    evidence = validate_m5_reference_dpo(arguments.evidence_root, project_root)
    write_json(arguments.output, evidence)
    print(f"M5 reference DPO validation passed; wrote {arguments.output}")


if __name__ == "__main__":
    main()
