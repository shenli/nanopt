"""Strict exact-token trajectory, metric, and summary records for GRPO/RLVR."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nanopt.core.advantages import AdvantageMode

TrajectoryFinishReason = Literal["eos", "protocol_stop", "max_length", "error"]
ParserRecordStatus = Literal["valid", "invalid", "error"]
VerifierRecordStatus = Literal["correct", "incorrect", "not_run", "error"]


class GrpoRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GrpoPromptRecord(GrpoRecord):
    messages: list[dict[str, str]]
    token_ids: list[int]
    attention_mask: list[int]


class GrpoCompletionRecord(GrpoRecord):
    completion_index: int = Field(ge=0)
    token_ids: list[int] = Field(min_length=1)
    action_mask: list[int] = Field(min_length=1)
    old_logprobs: list[float] = Field(min_length=1)
    reference_logprobs: list[float] | None = None
    decoded_text: str
    finish_reason: TrajectoryFinishReason
    reward: float
    reward_components: dict[str, float]
    advantage: float
    parser_status: ParserRecordStatus
    parsed_answer: str | None = None
    verifier_status: VerifierRecordStatus
    generation_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_token_coordinates(self) -> GrpoCompletionRecord:
        length = len(self.token_ids)
        if len(self.action_mask) != length or len(self.old_logprobs) != length:
            raise ValueError("token IDs, action mask, and old log probabilities must align")
        if self.reference_logprobs is not None and len(self.reference_logprobs) != length:
            raise ValueError("reference log probabilities must align with exact sampled token IDs")
        if any(value not in {0, 1} for value in self.action_mask):
            raise ValueError("action mask must be binary")
        if not any(self.action_mask):
            raise ValueError("a GRPO completion needs at least one active sampled token")
        return self


class GrpoTrajectoryRecord(GrpoRecord):
    schema_version: Literal[1] = 1
    trajectory_id: str
    run_id: str
    iteration: int = Field(ge=0)
    task_id: str
    prompt: GrpoPromptRecord
    group_reward_mean: float
    group_reward_std: float = Field(ge=0)
    advantage_mode: AdvantageMode
    completions: list[GrpoCompletionRecord] = Field(min_length=2)


class GrpoMetricRecord(GrpoRecord):
    schema_version: Literal[1] = 1
    run_id: str
    iteration: int = Field(ge=0)
    prompt_count: int = Field(gt=0)
    completion_count: int = Field(gt=1)
    active_tokens: int = Field(gt=0)
    reward_mean: float
    reward_std: float = Field(ge=0)
    reward_min: float
    reward_max: float
    correctness_rate: float = Field(ge=0, le=1)
    parser_success_rate: float = Field(ge=0, le=1)
    group_reward_std_mean: float = Field(ge=0)
    degenerate_group_fraction: float = Field(ge=0, le=1)
    advantage_mean: float
    advantage_std: float = Field(ge=0)
    advantage_max_abs: float = Field(ge=0)
    completion_length_mean: float = Field(gt=0)
    protocol_stop_fraction: float = Field(ge=0, le=1)
    eos_fraction: float = Field(ge=0, le=1)
    max_length_fraction: float = Field(ge=0, le=1)
    policy_loss: float
    kl_loss: float
    total_loss: float
    clip_fraction: float = Field(ge=0, le=1)
    ratio_mean: float = Field(gt=0)
    ratio_p95: float = Field(gt=0)
    current_minus_old_logp_mean: float
    sampled_action_surprisal: float = Field(ge=0)
    learning_rate: float = Field(ge=0)
    gradient_norm: float = Field(ge=0)
    gradient_clipped: bool
    rollout_seconds: float = Field(ge=0)
    training_seconds: float = Field(ge=0)
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)


class GrpoSummary(GrpoRecord):
    schema_version: Literal[1] = 1
    run_id: str
    iterations: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    trajectories: int = Field(gt=0)
    completions: int = Field(gt=1)
    mean_reward: float
    correctness_rate: float = Field(ge=0, le=1)
    parser_success_rate: float = Field(ge=0, le=1)
    degenerate_group_fraction: float = Field(ge=0, le=1)
    mean_clip_fraction: float = Field(ge=0, le=1)
    parent_dpo_adapter_sha256: str
    grpo_adapter_sha256: str
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
    representative: bool
    advantage_mode: AdvantageMode
    loss_normalization: Literal["token_mean", "sequence_mean"]
    clip_epsilon: float = Field(gt=0, lt=1)
    kl_beta: float = Field(ge=0)
    kl_estimator: Literal["direct", "k3"]
