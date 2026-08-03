from __future__ import annotations

from pathlib import Path

import pytest

from nanopt.config.loader import ConfigRepository
from nanopt.config.resolver import resolve_config
from nanopt.data.arithmetic import ArithmeticGeneratorConfig, generate_tasks
from nanopt.data.schemas import ArithmeticTask, SplitName
from nanopt.data.splits import SPLIT_ORDER, build_splits
from nanopt.eval.records import EvaluationResult
from nanopt.reporting.builder import build_evaluation_report
from nanopt.runtime.artifacts import append_jsonl, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata
from nanopt.runtime.run_context import create_run_context
from scripts.validate_m3_reference_smoke import validate_m3_reference_smoke


def _dataset(evidence_root: Path) -> tuple[list[ArithmeticTask], str]:
    config = ArithmeticGeneratorConfig(seed=5, count=7)
    counts: dict[SplitName, int] = {name: 1 for name in SPLIT_ORDER}
    splits, manifest = build_splits(
        generate_tasks(config), counts=counts, seed=6, generator_config=config
    )
    tasks = [task for name in SPLIT_ORDER for task in splits[name]]
    tasks_path = evidence_root / "data" / "tasks.jsonl"
    for task in tasks:
        append_jsonl(tasks_path, task.model_dump(mode="json", exclude_none=True))
    write_json(
        evidence_root / "data" / "dataset_manifest.json",
        manifest.model_dump(mode="json"),
    )
    return tasks, manifest.dataset_fingerprint


def _doctor(evidence_root: Path) -> None:
    write_json(
        evidence_root / "doctor.json",
        {
            "schema_version": 1,
            "status": "warning",
            "os": "linux",
            "architecture": "x86_64",
            "exit_code": 2,
            "python_version": "3.11.0",
            "pytorch_version": "fixture-torch",
            "dependencies": {},
            "cuda": {
                "available": True,
                "device_count": 1,
                "driver_version": "fixture-driver",
                "runtime_version": "fixture-cuda",
                "gpus": [
                    {
                        "index": 0,
                        "name": "NVIDIA GeForce RTX 4070 Ti SUPER",
                        "total_vram_bytes": 16 * 1024**3,
                        "free_vram_bytes": 15 * 1024**3,
                        "compute_capability": "8.9",
                        "bf16_supported": True,
                    }
                ],
            },
            "tf32_available": True,
            "huggingface_cache": "/fixture/cache",
            "docker": {
                "executable_found": False,
                "daemon_reachable": False,
                "version": None,
            },
            "profile": {
                "matched": True,
                "requested_id": "rtx_4070_ti_super_16gb",
                "support_status": "proposed_unvalidated",
                "reasons": [],
            },
            "messages": ["hardware profile is proposed_unvalidated"],
        },
    )


def _result(task: ArithmeticTask, checkpoint: str, index: int) -> EvaluationResult:
    return EvaluationResult(
        result_id=f"result-{checkpoint}-{index}",
        run_id="calibration" if checkpoint == "base-calibration" else "reference-base",
        checkpoint_id=checkpoint,
        task_id=task.task_id,
        split=task.split or "test_iid",
        sample_index=0,
        seed=index,
        generation_config_sha256="a" * 64,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3],
        response_text=f"<answer>{task.target.canonical_answer}</answer>",
        parser_status="valid",
        parsed_answer=task.target.canonical_answer,
        verifier_status="correct",
        reward_components={"correctness": 1.0},
        finish_reason="eos",
        generation_seconds=0.01,
    )


def _run(
    evidence_root: Path,
    project_root: Path,
    tasks: list[ArithmeticTask],
    dataset_fingerprint: str,
    *,
    run_id: str,
    checkpoint: str,
    representative: bool,
) -> None:
    result = resolve_config(
        repository=ConfigRepository(project_root / "configs"),
        hardware_id="rtx_4070_ti_super_16gb",
        model_id="qwen3_0_6b_base",
        experiment_id="base_eval",
    )
    context = create_run_context(
        result,
        artifacts_root=evidence_root / "runs",
        run_id=run_id,
        git_root=project_root,
    )
    profile = result.config.model
    context.manifest["git"] = {
        **collect_git_metadata(project_root),
        "dirty": False,
    }
    context.manifest["model"].update(
        {
            "resolved_revision": profile.source.revision,
            "tokenizer_revision": profile.source.tokenizer_revision,
            "chat_template_sha256": "b" * 64,
            "base_parameter_count": 1,
            "trainable_parameter_count": 0,
        }
    )
    context.manifest["data"]["fingerprints"] = {
        "dataset": dataset_fingerprint,
        "task_file_sha256": sha256_file(evidence_root / "data" / "tasks.jsonl"),
        "split_manifest_sha256": sha256_file(evidence_root / "data" / "dataset_manifest.json"),
    }
    context.manifest["evaluation"] = {
        "mode": "deterministic",
        "device": "cuda",
        "task_count": len(tasks),
        "representative": representative,
    }
    for index, task in enumerate(tasks):
        append_jsonl(
            context.run_dir / "samples.jsonl",
            _result(task, checkpoint, index).model_dump(mode="json"),
        )
    build_evaluation_report(context.run_dir)
    context.manifest["artifacts"] = [
        {"path": name, "kind": kind, "sha256": sha256_file(context.run_dir / name)}
        for name, kind in (
            ("samples.jsonl", "evaluation_examples"),
            ("summary.json", "evaluation_summary"),
            ("report.md", "markdown_report"),
            ("report.html", "html_report"),
        )
    ]
    context.set_status("completed")


def _bundle(tmp_path: Path, project_root: Path) -> Path:
    evidence_root = tmp_path / "m3-reference"
    tasks, fingerprint = _dataset(evidence_root)
    _doctor(evidence_root)
    selected = [
        task
        for task in tasks
        if task.split in {"test_iid", "test_compositional", "test_range", "test_format_attack"}
    ]
    _run(
        evidence_root,
        project_root,
        selected[:2],
        fingerprint,
        run_id="calibration",
        checkpoint="base-calibration",
        representative=False,
    )
    _run(
        evidence_root,
        project_root,
        selected,
        fingerprint,
        run_id="reference-base",
        checkpoint="base",
        representative=True,
    )
    return evidence_root


def test_m3_reference_validator_accepts_complete_fixture(
    tmp_path: Path, project_root: Path
) -> None:
    evidence_root = _bundle(tmp_path, project_root)

    summary = validate_m3_reference_smoke(evidence_root, project_root)

    assert summary["status"] == "m3_reference_smoke_passed"
    dataset = summary["dataset"]
    runs = summary["runs"]
    assert isinstance(dataset, dict)
    assert isinstance(runs, list)
    assert dataset["records"] == 7
    assert [run["task_count"] for run in runs] == [2, 4]


def test_m3_reference_validator_detects_artifact_tampering(
    tmp_path: Path, project_root: Path
) -> None:
    evidence_root = _bundle(tmp_path, project_root)
    report = evidence_root / "runs" / "reference-base" / "report.md"
    report.write_text(report.read_text() + "\ntampered\n")

    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        validate_m3_reference_smoke(evidence_root, project_root)
