from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanopt.eval.records import EvaluationResult
from nanopt.reporting.builder import UnsafeReportValue, build_evaluation_report
from nanopt.runtime.artifacts import append_jsonl


def _write_result(run_dir: Path, *, run_id: str = "run-1", checkpoint: str = "base") -> None:
    result = EvaluationResult(
        result_id="result-1",
        run_id=run_id,
        checkpoint_id=checkpoint,
        task_id="task-1",
        split="test_iid",
        sample_index=0,
        seed=7,
        generation_config_sha256="a" * 64,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3],
        response_text="<script>alert('not copied')</script> /Users/private token=hidden",
        parser_status="invalid",
        parsed_answer=None,
        verifier_status="incorrect",
        reward_components={"correctness": 0.0},
        finish_reason="length",
        generation_seconds=0.1,
    )
    append_jsonl(run_dir / "samples.jsonl", result.model_dump(mode="json"))


def test_report_builds_markdown_html_and_summary_from_fixture(tmp_path: Path) -> None:
    _write_result(tmp_path)

    artifacts = build_evaluation_report(tmp_path)

    markdown = (tmp_path / artifacts.markdown).read_text()
    html = (tmp_path / artifacts.html).read_text()
    summary = json.loads((tmp_path / artifacts.summary).read_text())
    assert "Exact-answer accuracy" in markdown
    assert 'href="samples.jsonl"' in html
    assert summary["examples"] == 1
    assert summary["pass_at_k"][0]["k"] == 1
    combined = markdown + html
    assert "/Users/private" not in combined
    assert "token=hidden" not in combined
    assert "<script>" not in combined


@pytest.mark.parametrize(
    ("run_id", "checkpoint"),
    [
        ("/Users/private/run", "base"),
        ("run", "gho_abcdefghijklmnopqrstuvwxyz"),
        ("../run", "base"),
    ],
)
def test_report_rejects_path_or_secret_bearing_identity(
    tmp_path: Path, run_id: str, checkpoint: str
) -> None:
    _write_result(tmp_path, run_id=run_id, checkpoint=checkpoint)
    with pytest.raises(UnsafeReportValue):
        build_evaluation_report(tmp_path)


def test_report_rejects_missing_or_empty_examples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        build_evaluation_report(tmp_path)
    (tmp_path / "samples.jsonl").touch()
    with pytest.raises(ValueError, match="no evaluation"):
        build_evaluation_report(tmp_path)


def test_report_rejects_mixed_checkpoint_aggregates(tmp_path: Path) -> None:
    _write_result(tmp_path, checkpoint="base")
    _write_result(tmp_path, checkpoint="sft")
    with pytest.raises(ValueError, match="exactly one checkpoint"):
        build_evaluation_report(tmp_path)
