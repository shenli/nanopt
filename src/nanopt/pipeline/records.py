"""Strict records for pipeline lineage, retries, and independently resumable stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PipelineRecord(BaseModel):
    """Reject unrecorded fields so the parent manifest remains a real contract."""

    model_config = ConfigDict(extra="forbid", strict=True)


class StageAttempt(PipelineRecord):
    """One retained attempt, successful or failed, for a logical pipeline stage."""

    attempt: int = Field(ge=1)
    run_id: str
    run_directory: str | None = None
    child_manifest_sha256: str | None = None
    artifact_sha256: str | None = None
    started_at: str
    finished_at: str
    wall_seconds: float = Field(ge=0)
    status: Literal["completed", "failed"]
    failure: str | None = None


class PipelineStage(PipelineRecord):
    """Logical stage state plus immutable input/output lineage."""

    id: str
    kind: Literal["calibration", "evaluation", "data", "training", "report"]
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    input_checkpoint_sha256: str | None = None
    output_checkpoint_sha256: str | None = None
    output_path: str | None = None
    attempts: list[StageAttempt] = Field(default_factory=list)


class FailureRetryRecord(PipelineRecord):
    """Compact disclosure copied from failed stage attempts."""

    stage_id: str
    attempt: int = Field(ge=1)
    failure: str
    retry_run_id: str | None = None


class PipelineManifest(PipelineRecord):
    """Parent contract joining source, data, children, checkpoints, and reports."""

    schema_version: Literal[1] = 1
    pipeline_run_id: str
    recipe_id: str
    status: Literal["created", "running", "completed", "failed"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    git: dict[str, str | bool | None]
    hardware_id: str
    model_id: str
    task_file_sha256: str
    split_manifest_sha256: str
    dataset_fingerprint: str
    protected_splits_used_for_training: Literal[False] = False
    stages: list[PipelineStage]
    failures_and_retries: list[FailureRetryRecord] = Field(default_factory=list)
    total_wall_seconds: float | None = Field(default=None, ge=0)
    phase_peak_reserved_bytes: dict[str, int] = Field(default_factory=dict)
    final_checkpoint_sha256: str | None = None
    comparison_artifacts: dict[str, str] = Field(default_factory=dict)
