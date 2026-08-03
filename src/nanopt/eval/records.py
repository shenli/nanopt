"""Typed example-level records written before evaluation aggregation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationResult(BaseModel):
    """One generated response and all evidence needed to rescore it.

    Token IDs are optional only for compatibility with external result imports. NanoPT's own
    baseline runner records them so the sampled sequence never needs to be reconstructed from text.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    result_id: str
    run_id: str
    checkpoint_id: str
    task_id: str
    split: str
    sample_index: int = Field(ge=0)
    seed: int
    generation_config_sha256: str
    prompt_token_ids: list[int] | None
    completion_token_ids: list[int] | None
    response_text: str
    parser_status: Literal["valid", "invalid", "error"]
    parsed_answer: str | None
    verifier_status: Literal["correct", "incorrect", "not_run", "error"]
    reward_components: dict[str, float]
    finish_reason: str | None
    generation_seconds: float | None = Field(default=None, ge=0)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
