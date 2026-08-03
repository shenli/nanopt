"""Build a dependency-free evaluation report from example-level JSONL records."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nanopt.eval.metrics import aggregate_results, pass_at_k_by_task
from nanopt.eval.records import EvaluationResult
from nanopt.runtime.artifacts import read_jsonl, write_json, write_text

_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+ -]{0,199}")
_SECRET_MARKERS = re.compile(
    r"(?:gh[opsu]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"(?:token|password|secret|api[_-]?key)\s*[:=])",
    flags=re.IGNORECASE,
)


class UnsafeReportValue(ValueError):
    """Raised when identity text could leak a path, secret, or markup into a report."""


@dataclass(frozen=True)
class ReportArtifacts:
    """Relative report paths suitable for recording in a run manifest."""

    markdown: Path
    html: Path
    summary: Path


def _public_label(value: str, *, name: str) -> str:
    if not _SAFE_LABEL.fullmatch(value):
        raise UnsafeReportValue(f"{name} contains unsupported or unsafe characters")
    if Path(value).is_absolute() or value.startswith(("~/", "..")):
        raise UnsafeReportValue(f"{name} must not contain an absolute or parent path")
    if _SECRET_MARKERS.search(value):
        raise UnsafeReportValue(f"{name} resembles secret-bearing text")
    return value


def _interval_line(metric: dict[str, Any]) -> str:
    return (
        f"{100 * float(metric['estimate']):.1f}% "
        f"({100 * float(metric['lower']):.1f}-{100 * float(metric['upper']):.1f}%, "
        f"n={int(metric['count'])}, Wilson 95% CI)"
    )


def _load_results(samples_path: Path) -> list[EvaluationResult]:
    try:
        return [
            EvaluationResult.model_validate(item, strict=True) for item in read_jsonl(samples_path)
        ]
    except ValidationError as exc:
        raise ValueError(f"invalid evaluation result in {samples_path.name}: {exc}") from exc


def _pass_metrics(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    sample_counts: dict[str, int] = {}
    for result in results:
        sample_counts[result.task_id] = sample_counts.get(result.task_id, 0) + 1
    if not sample_counts or len(set(sample_counts.values())) != 1:
        return []
    samples = next(iter(sample_counts.values()))
    metrics: list[dict[str, Any]] = []
    for k in sorted({1, samples}):
        metric = pass_at_k_by_task(results, k=k)
        metrics.append({"k": k, **metric.__dict__})
    return metrics


def _markdown(
    *,
    run_id: str,
    checkpoint_ids: list[str],
    summary: dict[str, Any],
    pass_metrics: list[dict[str, Any]],
) -> str:
    checkpoint_text = ", ".join(checkpoint_ids)
    lines = [
        "# NanoPT Evaluation Report",
        "",
        "> Smoke evidence only. This report does not claim validated hardware or pipeline support.",
        "",
        "## Identity",
        "",
        f"- Run: `{run_id}`",
        f"- Checkpoint(s): `{checkpoint_text}`",
        f"- Tasks: {summary['tasks']}",
        f"- Generated examples: {summary['examples']}",
        "",
        "## Headline metrics",
        "",
        "| Metric | Estimate and interval |",
        "| --- | --- |",
        f"| Exact-answer accuracy | {_interval_line(summary['accuracy'])} |",
        f"| Parse rate | {_interval_line(summary['parse_rate'])} |",
        f"| EOS fraction | {100 * float(summary['eos_fraction']):.1f}% |",
    ]
    for metric in pass_metrics:
        lines.append(
            f"| pass@{metric['k']} | {100 * float(metric['estimate']):.1f}% "
            f"(direct-first-{metric['k']} Wilson interval: "
            f"{100 * float(metric['lower']):.1f}-{100 * float(metric['upper']):.1f}%) |"
        )
    lengths = summary["completion_tokens"]
    lines.extend(
        [
            "",
            "## Completion length",
            "",
            f"Mean {float(lengths['mean']):.2f} tokens; range "
            f"{int(lengths['minimum'])}-{int(lengths['maximum'])} tokens.",
            "",
            "## Inspect the evidence",
            "",
            "Every aggregate above is rebuildable from [samples.jsonl](samples.jsonl). The JSONL "
            "contains exact prompt/completion token IDs, seeds, parser outcomes, verifier "
            "outcomes, finish reasons, and generation timing.",
            "",
            "Generated response bodies are intentionally not copied into this report. Keeping them "
            "in the example artifact prevents model output from injecting HTML, absolute local "
            "paths, or secret-like text into a shareable report.",
            "",
        ]
    )
    return "\n".join(lines)


def _html(markdown_values: dict[str, Any], pass_metrics: list[dict[str, Any]]) -> str:
    run_id = html.escape(str(markdown_values["run_id"]))
    checkpoints = html.escape(", ".join(markdown_values["checkpoint_ids"]))
    summary = markdown_values["summary"]
    pass_rows = "".join(
        "<tr><td>pass@{k}</td><td>{estimate:.1f}%</td><td>{lower:.1f}-{upper:.1f}%</td></tr>".format(
            k=int(metric["k"]),
            estimate=100 * float(metric["estimate"]),
            lower=100 * float(metric["lower"]),
            upper=100 * float(metric["upper"]),
        )
        for metric in pass_metrics
    )
    accuracy = summary["accuracy"]
    parse_rate = summary["parse_rate"]
    accuracy_estimate = 100 * float(accuracy["estimate"])
    accuracy_lower = 100 * float(accuracy["lower"])
    accuracy_upper = 100 * float(accuracy["upper"])
    parse_estimate = 100 * float(parse_rate["estimate"])
    parse_lower = 100 * float(parse_rate["lower"])
    parse_upper = 100 * float(parse_rate["upper"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NanoPT evaluation report</title>
  <style>
    body {{
      font: 16px/1.55 system-ui, sans-serif; max-width: 880px;
      margin: 3rem auto; padding: 0 1rem; color: #17212b;
    }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d6dce2; padding: .65rem; text-align: left; }}
    .notice {{ background: #f4f7fa; border-left: 4px solid #61758a; padding: .8rem 1rem; }}
    code {{ background: #eef1f4; padding: .1rem .3rem; }}
  </style>
</head>
<body>
  <h1>NanoPT Evaluation Report</h1>
  <p class="notice">
    Smoke evidence only. This report does not claim validated hardware or pipeline support.
  </p>
  <h2>Identity</h2>
  <p>Run <code>{run_id}</code>; checkpoint(s) <code>{checkpoints}</code>;<br>
     {int(summary["tasks"])} tasks and {int(summary["examples"])} generated examples.</p>
  <h2>Headline metrics</h2>
  <table><thead><tr><th>Metric</th><th>Estimate</th><th>Wilson 95% interval</th></tr></thead><tbody>
    <tr><td>Exact-answer accuracy</td><td>{accuracy_estimate:.1f}%</td>
      <td>{accuracy_lower:.1f}-{accuracy_upper:.1f}%</td></tr>
    <tr><td>Parse rate</td><td>{parse_estimate:.1f}%</td>
      <td>{parse_lower:.1f}-{parse_upper:.1f}%</td></tr>
    {pass_rows}
  </tbody></table>
  <h2>Inspect the evidence</h2>
  <p>Rebuild every aggregate from <a href="samples.jsonl">samples.jsonl</a>.
    Response bodies stay in that example artifact so model output cannot inject markup,
    local paths, or secret-like text into this shareable HTML file.</p>
</body>
</html>
"""


def build_evaluation_report(run_dir: Path) -> ReportArtifacts:
    """Rebuild ``summary.json``, ``report.md``, and self-contained ``report.html`` locally.

    Only a small whitelist of record fields enters the reports. Run/checkpoint labels must be safe
    relative identifiers. Environment values, absolute paths, prompts, and response text are never
    embedded. The resulting HTML has no scripts, external assets, or server dependency.
    """

    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    samples_path = run_dir / "samples.jsonl"
    if not samples_path.is_file():
        raise ValueError(f"missing evaluation examples: {samples_path.name}")
    results = _load_results(samples_path)
    if not results:
        raise ValueError("samples.jsonl contains no evaluation results")
    run_ids = {_public_label(item.run_id, name="run_id") for item in results}
    if len(run_ids) != 1:
        raise ValueError("report input mixes multiple run IDs")
    checkpoint_ids = sorted(
        {_public_label(item.checkpoint_id, name="checkpoint_id") for item in results}
    )
    if len(checkpoint_ids) != 1:
        raise ValueError("M3 reports require exactly one checkpoint per samples.jsonl")
    run_id = next(iter(run_ids))
    summary = aggregate_results(results)
    pass_metrics = _pass_metrics(results)
    summary["pass_at_k"] = pass_metrics
    summary["run_id"] = run_id
    summary["checkpoint_ids"] = checkpoint_ids

    summary_path = run_dir / "summary.json"
    markdown_path = run_dir / "report.md"
    html_path = run_dir / "report.html"
    write_json(summary_path, summary)
    write_text(
        markdown_path,
        _markdown(
            run_id=run_id,
            checkpoint_ids=checkpoint_ids,
            summary=summary,
            pass_metrics=pass_metrics,
        ),
    )
    write_text(
        html_path,
        _html(
            {"run_id": run_id, "checkpoint_ids": checkpoint_ids, "summary": summary},
            pass_metrics,
        ),
    )
    return ReportArtifacts(
        markdown=Path(markdown_path.name),
        html=Path(html_path.name),
        summary=Path(summary_path.name),
    )
