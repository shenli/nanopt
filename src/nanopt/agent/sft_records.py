"""Versioned, replayable records for Agent SFT datasets and runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from nanopt.agent.records import AgentRecord


class AgentChatMessage(AgentRecord):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class AgentSftExample(AgentRecord):
    """One exact-token action target derived from a retained environment transition."""

    schema_version: Literal[1] = 1
    example_id: str
    split: Literal["train", "validation"]
    example_kind: Literal["demonstration", "recovery"]
    task_id: str
    task_version: str
    trajectory_id: str
    source_trajectory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step_index: int = Field(ge=0)
    context_policy: Literal["observation_snapshot", "full_transcript"]
    messages: list[AgentChatMessage] = Field(min_length=2)
    completion: str = Field(min_length=1)
    target_action: dict[str, Any]
    input_ids: list[int] = Field(min_length=2)
    attention_mask: list[bool] = Field(min_length=2)
    action_mask: list[bool] = Field(min_length=2)
    prompt_length: int = Field(gt=0)
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def check_exact_token_coordinates(self) -> AgentSftExample:
        length = len(self.input_ids)
        if len(self.attention_mask) != length or len(self.action_mask) != length:
            raise ValueError("token IDs and masks must use the same full-sequence coordinates")
        if self.prompt_length >= length:
            raise ValueError("prompt_length must precede at least one completion token")
        if any(self.action_mask[: self.prompt_length]):
            raise ValueError("prompt tokens cannot be active action targets")
        if not any(self.action_mask[self.prompt_length :]):
            raise ValueError("example must contain at least one active action token")
        if not all(self.attention_mask):
            raise ValueError("stored examples are unpadded; attention_mask must be all true")
        return self


class AgentSftDatasetManifest(AgentRecord):
    schema_version: Literal[1] = 1
    dataset_id: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_id: str
    suite_version: str
    context_policy: Literal["observation_snapshot", "full_transcript"]
    tokenizer_revision: str
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    examples_file: str
    examples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_trajectories_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_tasks: list[str] = Field(min_length=1)
    validation_tasks: list[str] = Field(min_length=1)
    train_examples: int = Field(gt=0)
    validation_examples: int = Field(gt=0)
    demonstration_examples: int = Field(gt=0)
    recovery_examples: int = Field(ge=0)
    exact_replays_passed: int = Field(gt=0)
    source_trajectories: int = Field(gt=0)
    hidden_test_source_included: Literal[False] = False


class AgentSftSummary(AgentRecord):
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
    context_policy: Literal["observation_snapshot", "full_transcript"]
    held_out_task_ids: list[str] = Field(min_length=1)
