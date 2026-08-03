"""Validate an M3 reference smoke bundle without loading a language model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import jsonschema

from nanopt.config.loader import ConfigRepository
from nanopt.config.models import BaseEvalExperiment
from nanopt.eval.io import (
    read_arithmetic_tasks,
    read_split_manifest,
    validate_tasks_against_manifest,
)
from nanopt.eval.records import EvaluationResult
from nanopt.runtime.artifacts import read_jsonl, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata

_SECRET = re.compile(
    r"(?:gh[opsu]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"(?:password|secret|api[_-]?key)\s*[:=])",
    flags=re.IGNORECASE,
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_doctor(evidence_root: Path, project_root: Path) -> dict[str, Any]:
    doctor = _read_object(evidence_root / "doctor.json")
    doctor_schema = _read_object(project_root / "specs" / "schemas" / "doctor_report.schema.json")
    jsonschema.Draft202012Validator(doctor_schema).validate(doctor)
    _require(doctor.get("os") == "linux", "reference smoke requires Linux")
    _require(doctor.get("architecture") == "x86_64", "reference smoke requires x86_64")
    cuda = doctor.get("cuda")
    _require(isinstance(cuda, dict), "doctor report is missing CUDA details")
    _require(cuda.get("available") is True, "doctor did not find usable CUDA")
    _require(cuda.get("device_count") == 1, "reference smoke requires exactly one CUDA device")
    gpus = cuda.get("gpus")
    _require(isinstance(gpus, list) and len(gpus) == 1, "doctor must record exactly one GPU")
    gpu = gpus[0]
    _require(isinstance(gpu, dict), "doctor GPU record is malformed")
    _require(
        "RTX 4070 Ti SUPER" in str(gpu.get("name")),
        "GPU is not an RTX 4070 Ti SUPER",
    )
    _require(
        int(gpu.get("total_vram_bytes", 0)) >= 15 * 1024**3,
        "recorded GPU memory is below 15 GiB",
    )
    profile = doctor.get("profile")
    _require(isinstance(profile, dict), "doctor report is missing profile comparison")
    _require(profile.get("matched") is True, "doctor hardware profile did not match")
    _require(
        profile.get("requested_id") == "rtx_4070_ti_super_16gb",
        "doctor checked the wrong hardware profile",
    )
    _require(doctor.get("exit_code") in {0, 2}, "doctor reported an unusable environment")
    return doctor


def _validate_dataset(evidence_root: Path, project_root: Path) -> tuple[str, int]:
    tasks_path = evidence_root / "data" / "tasks.jsonl"
    manifest_path = evidence_root / "data" / "dataset_manifest.json"
    tasks = read_arithmetic_tasks(tasks_path)
    manifest = read_split_manifest(manifest_path)
    manifest_schema = _read_object(
        project_root / "specs" / "schemas" / "dataset_manifest.schema.json"
    )
    jsonschema.Draft202012Validator(manifest_schema).validate(manifest.model_dump(mode="json"))
    validate_tasks_against_manifest(tasks, manifest)
    return manifest.dataset_fingerprint, len(tasks)


def _validate_run(
    run_dir: Path,
    *,
    expected_representative: bool,
    expected_task_count: int,
    expected_checkpoint: str,
    expected_dataset_fingerprint: str,
    project_root: Path,
) -> dict[str, Any]:
    manifest = _read_object(run_dir / "run_manifest.json")
    run_schema = _read_object(project_root / "specs" / "schemas" / "run_manifest.schema.json")
    jsonschema.Draft202012Validator(run_schema).validate(manifest)
    _require(manifest.get("status") == "completed", f"run did not complete: {run_dir.name}")
    git = manifest.get("git")
    current_git = collect_git_metadata(project_root)
    _require(isinstance(git, dict), "run manifest is missing git identity")
    _require(git.get("dirty") is False, "reference smoke was run from a dirty checkout")
    _require(git.get("commit") == current_git["commit"], "run commit differs from checkout")

    evaluation = manifest.get("evaluation")
    _require(isinstance(evaluation, dict), "run manifest is missing evaluation evidence")
    _require(
        evaluation.get("representative") is expected_representative,
        f"unexpected representative label for {run_dir.name}",
    )
    _require(evaluation.get("mode") == "deterministic", "M3 smoke must be deterministic")
    _require(evaluation.get("device") == "cuda", "M3 smoke did not execute on CUDA")
    _require(
        evaluation.get("task_count") == expected_task_count,
        f"unexpected task count for {run_dir.name}",
    )

    fingerprints = manifest.get("data", {}).get("fingerprints", {})
    _require(
        fingerprints.get("dataset") == expected_dataset_fingerprint,
        f"dataset fingerprint mismatch in {run_dir.name}",
    )
    model = manifest.get("model")
    profile = ConfigRepository(project_root / "configs").model("qwen3_0_6b_base")
    _require(isinstance(model, dict), "run manifest is missing model identity")
    _require(
        model.get("resolved_revision") == profile.source.revision,
        "run used an unexpected model revision",
    )
    _require(
        model.get("tokenizer_revision") == profile.source.tokenizer_revision,
        "run used an unexpected tokenizer revision",
    )

    artifact_entries = manifest.get("artifacts")
    _require(isinstance(artifact_entries, list), "run manifest is missing artifact checksums")
    expected_artifacts = {"samples.jsonl", "summary.json", "report.md", "report.html"}
    _require(
        {entry.get("path") for entry in artifact_entries if isinstance(entry, dict)}
        == expected_artifacts,
        f"run artifact list is incomplete for {run_dir.name}",
    )
    for entry in artifact_entries:
        _require(isinstance(entry, dict), "run artifact entry is malformed")
        relative = Path(str(entry["path"]))
        _require(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path")
        _require(
            sha256_file(run_dir / relative) == entry.get("sha256"),
            f"artifact checksum mismatch: {relative}",
        )

    raw_results = read_jsonl(run_dir / "samples.jsonl")
    result_schema = _read_object(
        project_root / "specs" / "schemas" / "evaluation_result.schema.json"
    )
    _require(len(raw_results) == expected_task_count, "samples.jsonl has the wrong record count")
    results: list[EvaluationResult] = []
    for raw in raw_results:
        jsonschema.Draft202012Validator(result_schema).validate(raw)
        results.append(EvaluationResult.model_validate(raw, strict=True))
    _require(
        {result.checkpoint_id for result in results} == {expected_checkpoint},
        "samples contain an unexpected checkpoint identity",
    )
    _require(len({result.task_id for result in results}) == expected_task_count, "tasks repeat")

    summary = _read_object(run_dir / "summary.json")
    _require(summary.get("examples") == expected_task_count, "summary example count is wrong")
    report_text = (run_dir / "report.md").read_text(encoding="utf-8") + (
        run_dir / "report.html"
    ).read_text(encoding="utf-8")
    evidence_root = run_dir.parent.parent
    _require(str(evidence_root) not in report_text, "report leaks path")
    _require(_SECRET.search(report_text) is None, "report contains secret-like text")
    return {
        "run_id": manifest["run_id"],
        "task_count": expected_task_count,
        "representative": expected_representative,
        "samples_sha256": sha256_file(run_dir / "samples.jsonl"),
        "summary_sha256": sha256_file(run_dir / "summary.json"),
    }


def validate_m3_reference_smoke(evidence_root: Path, project_root: Path) -> dict[str, Any]:
    """Validate the complete two-run M3 bundle and return its compact evidence summary."""

    evidence_root = evidence_root.resolve()
    project_root = project_root.resolve()
    doctor = _validate_doctor(evidence_root, project_root)
    dataset_fingerprint, dataset_records = _validate_dataset(evidence_root, project_root)
    experiment = ConfigRepository(project_root / "configs").experiment("base_eval")
    if not isinstance(experiment, BaseEvalExperiment):
        raise ValueError("base_eval is not an evaluation experiment")
    selected_splits = set(experiment.data.splits)
    split_manifest = read_split_manifest(evidence_root / "data" / "dataset_manifest.json")
    reference_count = sum(
        count for split, count in split_manifest.counts.items() if split in selected_splits
    )
    calibration = _validate_run(
        evidence_root / "runs" / "calibration",
        expected_representative=False,
        expected_task_count=2,
        expected_checkpoint="base-calibration",
        expected_dataset_fingerprint=dataset_fingerprint,
        project_root=project_root,
    )
    reference = _validate_run(
        evidence_root / "runs" / "reference-base",
        expected_representative=True,
        expected_task_count=reference_count,
        expected_checkpoint="base",
        expected_dataset_fingerprint=dataset_fingerprint,
        project_root=project_root,
    )
    return {
        "schema_version": 1,
        "status": "m3_reference_smoke_passed",
        "git_commit": collect_git_metadata(project_root)["commit"],
        "hardware": {
            "profile": "rtx_4070_ti_super_16gb",
            "gpu": doctor["cuda"]["gpus"][0]["name"],
            "driver_version": doctor["cuda"]["driver_version"],
            "runtime_version": doctor["cuda"]["runtime_version"],
        },
        "dataset": {
            "fingerprint": dataset_fingerprint,
            "records": dataset_records,
        },
        "runs": [calibration, reference],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or arguments.evidence_root / "m3_smoke_evidence.json"
    summary = validate_m3_reference_smoke(arguments.evidence_root, arguments.project_root)
    write_json(output, summary)
    print(f"M3 reference smoke validation passed; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
