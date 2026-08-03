from __future__ import annotations

import json
from pathlib import Path

from nanopt.pipeline.report import build_pipeline_report
from nanopt.runtime.artifacts import write_json


def _evaluation(directory: Path, checkpoint: str, run_id: str) -> Path:
    directory.mkdir(parents=True)
    write_json(
        directory / "summary.json",
        {
            "examples": 2,
            "accuracy": {"estimate": 0.5, "lower": 0.1, "upper": 0.9},
            "parse_rate": {"estimate": 1.0},
        },
    )
    write_json(directory / "run_manifest.json", {"run_id": run_id})
    sample = {
        "result_id": f"{run_id}-result",
        "run_id": run_id,
        "checkpoint_id": checkpoint,
        "task_id": "task-1",
        "completion_token_ids": [1, 2],
        "response_text": "<answer>1</answer>",
        "verifier_status": "correct",
        "generation_seconds": 0.1,
    }
    (directory / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return directory


def test_report_repeat_comparison_excludes_run_identity(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    evaluations = {
        name: _evaluation(pipeline / name, "grpo" if name == "grpo" else name, name)
        for name in ("base", "sft", "dpo", "grpo")
    }
    repeat = _evaluation(pipeline / "repeat", "grpo", "repeat")
    training: dict[str, Path] = {}
    for name in ("sft", "dpo", "grpo"):
        directory = pipeline / f"train-{name}"
        directory.mkdir()
        write_json(
            directory / "summary.json",
            {"run_id": name, "peak_reserved_bytes": 100},
        )
        training[name] = directory

    build_pipeline_report(
        pipeline,
        evaluations=evaluations,
        training_runs=training,
        checkpoint_hashes={name: name * 64 for name in evaluations},
        repeat_evaluation=repeat,
    )

    comparison = json.loads((pipeline / "comparison.json").read_text())
    assert comparison["final_evaluation_repeat"]["exact_generation_match"] is True
