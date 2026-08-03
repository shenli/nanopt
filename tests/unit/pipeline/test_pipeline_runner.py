from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanopt.pipeline.records import PipelineManifest, PipelineStage
from nanopt.pipeline.run import PipelineRunner
from nanopt.runtime.artifacts import sha256_file


def _runner(tmp_path: Path) -> PipelineRunner:
    directory = tmp_path / "pipeline"
    directory.mkdir()
    return PipelineRunner(
        directory,
        PipelineManifest(
            pipeline_run_id="pipeline",
            recipe_id="math_pipeline",
            status="running",
            created_at="2026-08-03T00:00:00Z",
            git={"commit": "abc", "dirty": False},
            hardware_id="gpu",
            model_id="model",
            task_file_sha256="a" * 64,
            split_manifest_sha256="b" * 64,
            dataset_fingerprint="dataset",
            stages=[PipelineStage(id="data", kind="data")],
        ),
    )


def test_completed_artifact_stage_is_resumed_without_rerunning(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    calls = 0

    def action(_run_id: str) -> tuple[None, Path, str]:
        nonlocal calls
        calls += 1
        output = runner.directory / "data.json"
        output.write_text("stable", encoding="utf-8")
        return None, output, sha256_file(output)

    runner.run_stage("data", input_checkpoint_sha256=None, action=action)
    runner.run_stage("data", input_checkpoint_sha256=None, action=action)

    assert calls == 1
    assert json.loads(runner.manifest_path.read_text())["stages"][0]["status"] == "completed"


def test_failed_stage_is_retained_before_retry(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    def fail(_run_id: str) -> tuple[None, None, None]:
        raise RuntimeError("controlled failure")

    with pytest.raises(RuntimeError, match="controlled failure"):
        runner.run_stage("data", input_checkpoint_sha256=None, action=fail)

    output = runner.directory / "data.json"
    output.write_text("recovered", encoding="utf-8")
    runner.run_stage(
        "data",
        input_checkpoint_sha256=None,
        action=lambda run_id: (None, output, sha256_file(output)),
    )

    stage = runner.stage("data")
    assert [attempt.status for attempt in stage.attempts] == ["failed", "completed"]
    assert stage.attempts[-1].run_id == "data-retry-2"
    assert runner.manifest.failures_and_retries[0].failure.endswith("controlled failure")
