"""Create inspectable run directories before expensive work begins."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nanopt.config.provenance import serialize_provenance
from nanopt.config.resolver import ResolutionResult
from nanopt.runtime.artifacts import canonical_json, sha256_bytes, write_json, write_yaml
from nanopt.runtime.environment import collect_environment, collect_git_metadata

RunStatus = Literal["created", "running", "completed", "failed", "interrupted", "tainted"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def make_run_id(stage: str, config: dict[str, Any], now: datetime | None = None) -> str:
    """Create a path-safe run ID with a short deterministic config hash."""

    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    safe_stage = re.sub(r"[^a-z0-9_-]+", "-", stage.lower()).strip("-")
    if not safe_stage:
        raise ValueError("stage must contain at least one path-safe character")
    digest = sha256_bytes(canonical_json(config))[:10]
    return f"{timestamp}_{safe_stage}_{digest}"


def _dataset_ids(result: ResolutionResult) -> list[str]:
    """Extract either a training dataset or an agent task suite from the experiment union."""

    experiment = result.config.experiment.model_dump(mode="python")
    data = experiment.get("data")
    if isinstance(data, dict) and isinstance(data.get("dataset"), str):
        return [data["dataset"]]
    tasks = experiment.get("tasks")
    if isinstance(tasks, dict) and isinstance(tasks.get("suite"), str):
        return [tasks["suite"]]
    return []


@dataclass
class RunContext:
    """Mutable lifecycle facade whose on-disk manifest is atomically replaced."""

    run_dir: Path
    manifest: dict[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    def set_status(
        self,
        status: RunStatus,
        *,
        failure: dict[str, Any] | None = None,
    ) -> None:
        """Transition lifecycle timestamps and persist the complete manifest atomically."""

        self.manifest["status"] = status
        if status == "running" and self.manifest["started_at"] is None:
            self.manifest["started_at"] = utc_now()
        if status in {"completed", "failed", "interrupted", "tainted"}:
            self.manifest["finished_at"] = utc_now()
        self.manifest["failure"] = failure
        write_json(self.manifest_path, self.manifest)


def create_run_context(
    result: ResolutionResult,
    *,
    artifacts_root: Path = Path("artifacts/runs"),
    run_id: str | None = None,
    git_root: Path | None = None,
) -> RunContext:
    """Create the minimum run artifact contract before model or GPU allocation."""

    config_value = result.config.model_dump(mode="json", exclude_none=False)
    stage = result.config.experiment.stage
    selected_run_id = run_id or make_run_id(stage, config_value)
    if Path(selected_run_id).name != selected_run_id or selected_run_id in {".", ".."}:
        raise ValueError("run_id must be a single path-safe component")
    run_dir = artifacts_root / selected_run_id
    # exist_ok=False turns an accidental run-ID collision into a visible error instead of
    # mixing metrics and checkpoints from two runs.
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("checkpoints", "cache", "plots"):
        (run_dir / directory).mkdir()

    resolved_path = run_dir / "resolved_config.yaml"
    provenance_path = run_dir / "config_provenance.yaml"
    environment_path = run_dir / "environment.json"
    write_yaml(resolved_path, config_value)
    write_yaml(provenance_path, serialize_provenance(result.provenance))
    write_json(environment_path, collect_environment())

    # The manifest is created before a model is loaded. Fields that require expensive work
    # start as null/unresolved and are filled by later milestones as evidence becomes known.
    source = result.config.model.source
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": selected_run_id,
        "pipeline_run_id": None,
        "parent_run_ids": [],
        "stage": stage,
        "status": "created",
        "created_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "git": collect_git_metadata(git_root),
        "config": {
            "resolved_file": resolved_path.name,
            "sha256": sha256_bytes(resolved_path.read_bytes()),
            "provenance_file": provenance_path.name,
        },
        "environment_file": environment_path.name,
        "model": {
            "model_id": source.model_id,
            "resolved_revision": source.revision or "unresolved",
            "tokenizer_revision": source.tokenizer_revision or source.revision or "unresolved",
            "chat_template_sha256": None,
            "base_parameter_count": None,
            "trainable_parameter_count": None,
            "adapter_name": None,
            "adapter_sha256": None,
        },
        "data": {
            "dataset_ids": _dataset_ids(result),
            "fingerprints": {},
            "protected_splits_used_for_training": False,
        },
        "checkpoint": None,
        "artifacts": [],
        "failure": None,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    for filename in ("metrics.jsonl", "events.jsonl", "samples.jsonl"):
        (run_dir / filename).touch()
    return RunContext(run_dir=run_dir, manifest=manifest)
