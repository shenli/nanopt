from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nanopt.cli import app
from nanopt.eval.records import EvaluationResult
from nanopt.runtime.artifacts import append_jsonl
from nanopt.runtime.doctor import (
    CudaStatus,
    DockerStatus,
    DoctorReport,
    ProfileMatch,
)

runner = CliRunner()


def test_help_and_version_do_not_download_models() -> None:
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["--version"])
    assert help_result.exit_code == 0
    assert "config" in help_result.stdout
    assert "data" in help_result.stdout
    assert "doctor" in help_result.stdout
    assert "eval" in help_result.stdout
    assert "report" in help_result.stdout
    assert "train" in help_result.stdout
    assert "calibrate" in help_result.stdout
    assert version_result.exit_code == 0
    assert "0.1.0.dev0" in version_result.stdout


def test_config_resolve_writes_config_and_provenance(tmp_path: Path, project_root: Path) -> None:
    output = tmp_path / "resolved.yaml"
    result = runner.invoke(
        app,
        [
            "config",
            "resolve",
            "--config-dir",
            str(project_root / "configs"),
            "--experiment",
            "math_grpo",
            "--set",
            "rollout.group_size=2",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert output.is_file()
    assert output.with_name("resolved.provenance.yaml").is_file()
    assert "cli_override" in output.with_name("resolved.provenance.yaml").read_text()
    assert "group_size: 2" in output.read_text()


def test_config_resolve_reports_invalid_override(tmp_path: Path, project_root: Path) -> None:
    result = runner.invoke(
        app,
        [
            "config",
            "resolve",
            "--config-dir",
            str(project_root / "configs"),
            "--set",
            "data.not_a_field=1",
            "--output",
            str(tmp_path / "nope.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert "unknown override path" in result.stdout


def test_doctor_writes_json_even_when_environment_is_unusable(
    monkeypatch: object, tmp_path: Path, project_root: Path
) -> None:
    report = DoctorReport(
        status="unusable",
        exit_code=3,
        os="test",
        architecture="test",
        python_version="3.11.0",
        pytorch_version=None,
        dependencies={},
        cuda=CudaStatus(
            available=False,
            runtime_version=None,
            driver_version=None,
            device_count=0,
            gpus=[],
        ),
        tf32_available=False,
        huggingface_cache="cache",
        docker=DockerStatus(executable_found=False, daemon_reachable=False, version=None),
        profile=ProfileMatch(
            requested_id="rtx_4070_ti_super_16gb",
            matched=False,
            support_status="proposed_unvalidated",
            reasons=["fixture"],
        ),
        messages=["fixture"],
    )
    # pytest's MonkeyPatch is intentionally not imported into runtime package code.
    monkeypatch.setattr("nanopt.cli.collect_doctor_report", lambda *_args, **_kwargs: report)  # type: ignore[attr-defined]
    output = tmp_path / "doctor.json"
    result = runner.invoke(
        app,
        [
            "doctor",
            "--config-dir",
            str(project_root / "configs"),
            "--json",
            str(output),
        ],
    )
    assert result.exit_code == 3
    assert json.loads(output.read_text())["status"] == "unusable"


def test_report_build_command_uses_local_example_artifacts(tmp_path: Path) -> None:
    result = EvaluationResult(
        result_id="fixture-result",
        run_id="fixture-run",
        checkpoint_id="base",
        task_id="fixture-task",
        split="test_iid",
        sample_index=0,
        seed=1,
        generation_config_sha256="a" * 64,
        prompt_token_ids=[1],
        completion_token_ids=[2],
        response_text="<answer>1</answer>",
        parser_status="valid",
        parsed_answer="1",
        verifier_status="correct",
        reward_components={"correctness": 1.0},
        finish_reason="eos",
        generation_seconds=0.01,
    )
    append_jsonl(tmp_path / "samples.jsonl", result.model_dump(mode="json"))

    command = runner.invoke(app, ["report", "build", str(tmp_path)])

    assert command.exit_code == 0, command.stdout
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_data_generate_command_reproduces_task_file(tmp_path: Path, project_root: Path) -> None:
    common = [
        "data",
        "generate",
        "--generator-config",
        str(project_root / "tasks/arithmetic/generator_config.yaml"),
        "--split-config",
        str(project_root / "tasks/arithmetic/split_config.yaml"),
    ]
    first = tmp_path / "first.jsonl"
    command = runner.invoke(app, [*common, "--output", str(first)])
    assert command.exit_code == 0, command.stdout
    assert len(first.read_text().splitlines()) == 128
    assert (tmp_path / "dataset_manifest.json").is_file()

    second = tmp_path / "second.jsonl"
    repeated = runner.invoke(
        app,
        [
            *common,
            "--output",
            str(second),
            "--manifest",
            str(tmp_path / "second-manifest.json"),
        ],
    )
    assert repeated.exit_code == 0, repeated.stdout
    assert first.read_bytes() == second.read_bytes()

    same_path = tmp_path / "same.jsonl"
    rejected = runner.invoke(
        app,
        [*common, "--output", str(same_path), "--manifest", str(same_path)],
    )
    assert rejected.exit_code == 1
    assert not same_path.exists()
