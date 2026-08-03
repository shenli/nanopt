"""Strict public records for MiniSWE tasks, actions, observations, and trajectories."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

ToolName = Literal["list_files", "read_file", "search", "apply_patch", "run_tests", "finish"]
FinishReason = Literal[
    "model_finish",
    "budget_exhausted",
    "timeout",
    "sandbox_failure",
    "policy_failure",
    "error",
]


class AgentRecord(BaseModel):
    """Forbid accidental fields at every model/environment trust boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class AgentTaskBudgets(AgentRecord):
    tool_calls: int = Field(gt=0)
    test_runs: int = Field(gt=0)
    turns: int = Field(gt=0)
    wall_clock_seconds: int = Field(gt=0)


class AgentTaskCard(AgentRecord):
    """Versioned trusted metadata; hidden paths never enter model observations."""

    schema_version: Literal[1] = 1
    id: str
    version: str
    split: Literal["smoke", "reference"]
    issue: str
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    editable_globs: list[str]
    protected_globs: list[str]
    public_test_command: list[str]
    hidden_test_command: list[str]
    public_tests_total: int = Field(gt=0)
    hidden_tests_total: int = Field(gt=0)
    budgets: AgentTaskBudgets
    license: Literal["Apache-2.0"]


class AgentSuite(AgentRecord):
    schema_version: Literal[1] = 1
    id: str
    version: str
    tasks: list[str]


class ListFilesArguments(AgentRecord):
    path: str = "."
    max_depth: int = Field(default=4, ge=0, le=8)


class ReadFileArguments(AgentRecord):
    path: str
    start_line: int = Field(default=1, gt=0)
    end_line: int = Field(default=200, gt=0)


class SearchArguments(AgentRecord):
    query: str = Field(min_length=1, max_length=256)
    path: str = "."
    glob: str = "*.py"


class ApplyPatchArguments(AgentRecord):
    patch: str = Field(min_length=1, max_length=65536)


class RunTestsArguments(AgentRecord):
    """Intentionally empty: the trusted task chooses the command."""


class FinishArguments(AgentRecord):
    summary: str = Field(min_length=1, max_length=2000)


class ListFilesAction(AgentRecord):
    tool: Literal["list_files"]
    arguments: ListFilesArguments


class ReadFileAction(AgentRecord):
    tool: Literal["read_file"]
    arguments: ReadFileArguments


class SearchAction(AgentRecord):
    tool: Literal["search"]
    arguments: SearchArguments


class ApplyPatchAction(AgentRecord):
    tool: Literal["apply_patch"]
    arguments: ApplyPatchArguments


class RunTestsAction(AgentRecord):
    tool: Literal["run_tests"]
    arguments: RunTestsArguments


class FinishAction(AgentRecord):
    tool: Literal["finish"]
    arguments: FinishArguments


AgentAction = Annotated[
    ListFilesAction
    | ReadFileAction
    | SearchAction
    | ApplyPatchAction
    | RunTestsAction
    | FinishAction,
    Field(discriminator="tool"),
]
ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)


def parse_action(response: str, *, maximum_bytes: int = 65536) -> AgentAction:
    """Parse exactly one typed JSON action; never interpret text as a shell command."""

    if len(response.encode("utf-8")) > maximum_bytes:
        raise ValueError("model action exceeds the maximum byte count")
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model action is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("model action must be one JSON object")
    return ACTION_ADAPTER.validate_python(value, strict=True)


class BudgetState(AgentRecord):
    turns_remaining: int = Field(ge=0)
    tool_calls_remaining: int = Field(ge=0)
    test_runs_remaining: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    wall_clock_limit_seconds: int = Field(gt=0)


class ToolResult(AgentRecord):
    status: Literal["ok", "error"]
    code: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False


class AgentObservation(AgentRecord):
    schema_version: Literal[1] = 1
    task_id: str
    task_version: str
    issue: str
    allowed_tools: list[ToolName]
    budgets: BudgetState
    last_tool_result: ToolResult | None = None
    transcript: list[dict[str, Any]] = Field(default_factory=list)


class TestSummary(AgentRecord):
    status: Literal["passed", "failed", "timeout", "sandbox_error"]
    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    exit_code: int | None
    duration_seconds: float = Field(ge=0)
    output: str | None = None
    output_truncated: bool = False
    workspace_sha256: str


class AgentVerification(AgentRecord):
    public: TestSummary
    hidden: TestSummary
    final_score: float = Field(ge=0, le=1)
    policy_violation_penalty: float = Field(ge=0)


class AgentPolicyIdentity(AgentRecord):
    name: str
    version: str
    checkpoint_id: str | None
    generation: dict[str, str | int | float | bool | None]


class AgentStep(AgentRecord):
    step_index: int = Field(ge=0)
    observation: dict[str, Any]
    model_response: str
    model_token_ids: list[int] | None = None
    action_parse_status: Literal["valid", "invalid", "error"]
    action: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    budget_before: dict[str, Any]
    budget_after: dict[str, Any]
    workspace_state_sha256: str | None = None
    model_seconds: float | None = Field(default=None, ge=0)
    tool_seconds: float | None = Field(default=None, ge=0)
    policy_violations: list[str] = Field(default_factory=list)


class AgentTrajectory(AgentRecord):
    schema_version: Literal[1] = 1
    trajectory_id: str
    run_id: str
    task_id: str
    task_version: str
    environment_version: str
    model_checkpoint_id: str | None
    policy: AgentPolicyIdentity
    initial_snapshot_sha256: str
    steps: list[AgentStep]
    finish_reason: FinishReason
    final_workspace_sha256: str | None
    verification: AgentVerification


class AgentRunSummary(AgentRecord):
    schema_version: Literal[1] = 1
    run_id: str
    backend: Literal["fake", "docker"]
    policy: str
    tasks: int = Field(gt=0)
    solved: int = Field(ge=0)
    mean_score: float = Field(ge=0, le=1)
    policy_violations: int = Field(ge=0)
    total_steps: int = Field(ge=0)
    wall_seconds: float = Field(ge=0)
    representative: bool
    environment_trains_model: Literal[False] = False
