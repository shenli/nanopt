from __future__ import annotations

import pytest
from pydantic import ValidationError

from nanopt.pipeline.records import PipelineManifest, PipelineStage


def test_pipeline_manifest_rejects_unrecorded_fields() -> None:
    with pytest.raises(ValidationError):
        PipelineManifest.model_validate(
            {
                "pipeline_run_id": "pipeline",
                "recipe_id": "math_pipeline",
                "status": "created",
                "created_at": "2026-08-03T00:00:00Z",
                "git": {"commit": "abc", "dirty": False},
                "hardware_id": "gpu",
                "model_id": "model",
                "task_file_sha256": "a" * 64,
                "split_manifest_sha256": "b" * 64,
                "dataset_fingerprint": "dataset",
                "stages": [PipelineStage(id="base_eval", kind="evaluation")],
                "mystery": True,
            }
        )


def test_pipeline_stage_starts_pending() -> None:
    stage = PipelineStage(id="sft", kind="training")
    assert stage.status == "pending"
    assert stage.attempts == []
