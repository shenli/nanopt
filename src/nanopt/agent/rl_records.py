"""Strict exact-token records for short-horizon MiniSWE Agent RL."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from nanopt.agent.records import AgentRecord, FinishReason
from nanopt.core.advantages import AdvantageMode


class AgentRlAction(AgentRecord):
    """One sampled tool action in the token coordinates used during generation.

    ``prompt_token_ids`` are the exact online prompt. ``sampled_token_ids``, ``action_mask``, and
    ``old_logprobs`` share generated-token coordinates. The trainer concatenates the stored prompt
    and action IDs directly; decoded text is retained for inspection but is never tokenized again.
    """

    schema_version: Literal[1] = 1
    turn_index: int = Field(ge=0)
    prompt_token_ids: list[int] = Field(min_length=1)
    sampled_token_ids: list[int] = Field(min_length=1)
    action_mask: list[bool] = Field(min_length=1)
    old_logprobs: list[float] = Field(min_length=1)
    reference_logprobs: list[float] | None = None
    decoded_text: str
    action_parse_status: Literal["valid", "invalid", "error"]
    tool: str | None
    advantage: float = 0.0

    @model_validator(mode="after")
    def validate_exact_coordinates(self) -> AgentRlAction:
        length = len(self.sampled_token_ids)
        if len(self.action_mask) != length or len(self.old_logprobs) != length:
            raise ValueError(
                "Agent RL token IDs, action mask, and old log probabilities must align"
            )
        if self.reference_logprobs is not None and len(self.reference_logprobs) != length:
            raise ValueError("Agent RL reference log probabilities must align with sampled IDs")
        if not any(self.action_mask):
            raise ValueError("Agent RL action must contain at least one active sampled token")
        values = [*self.old_logprobs, *(self.reference_logprobs or [])]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Agent RL log probabilities must be finite")
        return self


class AgentRlEpisode(AgentRecord):
    """One independently reset MiniSWE episode and its terminal hidden reward."""

    schema_version: Literal[1] = 1
    episode_id: str
    run_id: str
    iteration: int = Field(ge=0)
    collected_policy_version: int = Field(ge=0)
    task_id: str
    task_version: str
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rollout_index: int = Field(ge=0)
    actions: list[AgentRlAction] = Field(min_length=1)
    finish_reason: FinishReason
    hidden_outcome_reward: float = Field(ge=0, le=1)
    hidden_passed: int = Field(ge=0)
    hidden_total: int = Field(gt=0)
    public_passed: bool
    policy_violations: int = Field(ge=0)
    advantage: float = 0.0

    @model_validator(mode="after")
    def validate_turns_and_hidden_summary(self) -> AgentRlEpisode:
        if [action.turn_index for action in self.actions] != list(range(len(self.actions))):
            raise ValueError("Agent RL action turn indices must be contiguous from zero")
        if self.hidden_passed > self.hidden_total:
            raise ValueError("hidden passed count cannot exceed hidden total")
        return self


class AgentRlGroup(AgentRecord):
    """Grouped episodes that all begin at one immutable task snapshot."""

    schema_version: Literal[1] = 1
    group_id: str
    run_id: str
    iteration: int = Field(ge=0)
    policy_version: int = Field(ge=0)
    task_id: str
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reward_mean: float
    reward_std: float = Field(ge=0)
    advantage_mode: AdvantageMode
    degenerate: bool
    episodes: list[AgentRlEpisode] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_identical_snapshot_group(self) -> AgentRlGroup:
        for episode in self.episodes:
            if episode.task_id != self.task_id or episode.snapshot_sha256 != self.snapshot_sha256:
                raise ValueError("Agent RL group episodes must share task and initial snapshot")
            if episode.iteration != self.iteration:
                raise ValueError("Agent RL group episodes must share collection iteration")
            if episode.collected_policy_version != self.policy_version:
                raise ValueError("Agent RL group episodes must share collection policy version")
        if sorted(episode.rollout_index for episode in self.episodes) != list(
            range(len(self.episodes))
        ):
            raise ValueError("Agent RL rollout indices must be contiguous from zero")
        return self


class AgentRlMetric(AgentRecord):
    schema_version: Literal[1] = 1
    run_id: str
    iteration: int = Field(ge=0)
    policy_version_before: int = Field(ge=0)
    policy_version_after: int = Field(gt=0)
    episodes: int = Field(gt=1)
    actions: int = Field(gt=0)
    active_tokens: int = Field(gt=0)
    reward_mean: float = Field(ge=0, le=1)
    reward_std: float = Field(ge=0)
    solved_rate: float = Field(ge=0, le=1)
    action_validity_rate: float = Field(ge=0, le=1)
    degenerate_group: bool
    policy_loss: float
    kl_loss: float
    total_loss: float
    clip_fraction: float = Field(ge=0, le=1)
    ratio_mean: float = Field(gt=0)
    gradient_norm: float = Field(ge=0)
    optimizer_steps: int = Field(gt=0)
    rollout_seconds: float = Field(ge=0)
    training_seconds: float = Field(ge=0)
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)


class AgentRlStalenessPoint(AgentRecord):
    label: Literal["fresh", "stale"]
    collected_policy_version: int = Field(ge=0)
    scored_policy_version: int = Field(ge=0)
    policy_lag: int = Field(ge=0)
    active_tokens: int = Field(gt=0)
    mean_abs_log_ratio: float = Field(ge=0)
    max_abs_log_ratio: float = Field(ge=0)
    approximate_ess_fraction: float = Field(gt=0, le=1)
    used_for_update: Literal[False] = False


class AgentRlStalenessStudy(AgentRecord):
    schema_version: Literal[1] = 1
    final_policy_version: int = Field(gt=0)
    fresh: AgentRlStalenessPoint
    stale: AgentRlStalenessPoint


class AgentRlCreditStudy(AgentRecord):
    schema_version: Literal[1] = 1
    episodes: int = Field(gt=0)
    successful_episodes: int = Field(ge=0)
    all_actions_active_tokens: int = Field(gt=0)
    terminal_action_active_tokens: int = Field(gt=0)
    terminal_token_fraction: float = Field(gt=0, le=1)
    active_tokens_by_tool: dict[str, int]


class AgentRlBudgetPoint(AgentRecord):
    checkpoint: Literal["reference", "agent_rl"]
    tool_budget: int = Field(gt=0)
    tasks: int = Field(gt=0)
    solved: int = Field(ge=0)
    mean_hidden_outcome_reward: float = Field(ge=0, le=1)
    action_validity_rate: float = Field(ge=0, le=1)


class AgentRlBudgetStudy(AgentRecord):
    schema_version: Literal[1] = 1
    task_ids: list[str] = Field(min_length=1)
    points: list[AgentRlBudgetPoint] = Field(min_length=4)


class AgentRlSummary(AgentRecord):
    schema_version: Literal[1] = 1
    run_id: str
    iterations: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    groups: int = Field(gt=0)
    episodes: int = Field(gt=1)
    actions: int = Field(gt=0)
    exact_sampled_tokens: Literal[True] = True
    hidden_reward_exposed_during_rollout: Literal[False] = False
    maximum_training_policy_lag: Literal[0] = 0
    mean_reward: float = Field(ge=0, le=1)
    action_validity_rate: float = Field(ge=0, le=1)
    degenerate_group_fraction: float = Field(ge=0, le=1)
    initial_validation_reward: float = Field(ge=0, le=1)
    final_validation_reward: float = Field(ge=0, le=1)
    parent_agent_sft_adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_rl_adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
    representative: bool
