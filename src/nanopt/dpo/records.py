"""Strict metric and summary records for inspectable DPO runs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DpoRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DpoMetricRecord(DpoRecord):
    schema_version: Literal[1] = 1
    run_id: str
    split: Literal["train", "validation"]
    optimizer_step: int = Field(ge=0)
    dpo_loss: float = Field(ge=0)
    policy_chosen_logp: float
    policy_rejected_logp: float
    policy_margin: float
    reference_margin: float
    implicit_reward_margin: float
    preference_accuracy: float = Field(ge=0, le=1)
    reward_accuracy: float = Field(ge=0, le=1)
    pair_count: int = Field(gt=0)
    chosen_active_tokens: float = Field(gt=0)
    rejected_active_tokens: float = Field(gt=0)
    learning_rate: float | None = Field(default=None, ge=0)
    gradient_norm: float | None = Field(default=None, ge=0)
    gradient_clipped: bool | None = None
    peak_allocated_bytes: int = Field(default=0, ge=0)
    peak_reserved_bytes: int = Field(default=0, ge=0)


class DpoSummary(DpoRecord):
    schema_version: Literal[1] = 1
    run_id: str
    optimizer_steps: int = Field(gt=0)
    train_pairs: int = Field(gt=0)
    validation_pairs: int = Field(gt=0)
    initial_validation_loss: float = Field(ge=0)
    final_validation_loss: float = Field(ge=0)
    initial_validation_policy_margin: float
    final_validation_policy_margin: float
    initial_validation_reward_accuracy: float = Field(ge=0, le=1)
    final_validation_reward_accuracy: float = Field(ge=0, le=1)
    reference_cache_sha256: str
    reference_cache_parity_max_abs_error: float = Field(ge=0)
    sft_adapter_sha256: str
    dpo_adapter_sha256: str
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
    representative: bool
