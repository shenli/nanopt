"""Strict SFT metric and summary records written by the training loop."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SftRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SftMetricRecord(SftRecord):
    schema_version: Literal[1] = 1
    run_id: str
    split: Literal["train", "validation"]
    optimizer_step: int = Field(ge=0)
    completion_nll: float = Field(ge=0)
    completion_token_accuracy: float = Field(ge=0, le=1)
    active_tokens: int = Field(gt=0)
    learning_rate: float | None = Field(default=None, ge=0)
    gradient_norm: float | None = Field(default=None, ge=0)
    gradient_clipped: bool | None = None
    tokens_per_second: float | None = Field(default=None, ge=0)
    peak_allocated_bytes: int = Field(default=0, ge=0)
    peak_reserved_bytes: int = Field(default=0, ge=0)


class SftSummary(SftRecord):
    schema_version: Literal[1] = 1
    run_id: str
    optimizer_steps: int = Field(gt=0)
    train_examples: int = Field(gt=0)
    validation_examples: int = Field(gt=0)
    initial_validation_nll: float = Field(ge=0)
    final_validation_nll: float = Field(ge=0)
    initial_validation_token_accuracy: float = Field(ge=0, le=1)
    final_validation_token_accuracy: float = Field(ge=0, le=1)
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
    representative: bool
