"""Explicit MiniSWE reset/step/finalize state machine."""

from __future__ import annotations

import difflib
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from nanopt.agent.policy import AgentPolicy
from nanopt.agent.records import (
    AgentObservation,
    AgentStep,
    AgentTrajectory,
    BudgetState,
    FinishAction,
    FinishReason,
    ToolName,
    ToolResult,
    parse_action,
)
from nanopt.agent.sandbox.base import SandboxBackend, SandboxLimits
from nanopt.agent.tasks import LoadedAgentTask, copy_snapshot
from nanopt.agent.verifier import (
    HiddenVerifier,
    protected_file_hashes,
    protected_files_unchanged,
)
from nanopt.agent.workspace import SafeWorkspace, WorkspacePolicyError, workspace_sha256
from nanopt.runtime.artifacts import canonical_json, sha256_bytes

ENVIRONMENT_VERSION = "mini-swe-environment-v1"
SECURITY_VIOLATION_CODES = {
    "invalid_path",
    "path_traversal",
    "symlink_escape",
    "symlink_forbidden",
    "protected_path",
    "path_not_editable",
    "binary_patch",
    "invalid_patch",
}


class MiniSWEEnvironment:
    """Own observable state, budgets, trusted tools, termination, and hidden verification."""

    def __init__(
        self,
        task: LoadedAgentTask,
        backend: SandboxBackend,
        *,
        run_id: str,
        allowed_tools: list[ToolName],
        limits: SandboxLimits,
        turn_limit: int | None = None,
        tool_call_limit: int | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if turn_limit is not None and turn_limit <= 0:
            raise ValueError("turn_limit must be positive")
        if tool_call_limit is not None and tool_call_limit <= 0:
            raise ValueError("tool_call_limit must be positive")
        self.task = task
        self.backend = backend
        self.run_id = run_id
        self.allowed_tools = allowed_tools
        self.limits = limits
        self.clock = clock
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.workspace_root: Path | None = None
        self.workspace: SafeWorkspace | None = None
        self.started_at = 0.0
        self.turns_remaining = min(task.card.budgets.turns, turn_limit or task.card.budgets.turns)
        self.tool_calls_remaining = min(
            task.card.budgets.tool_calls,
            tool_call_limit or task.card.budgets.tool_calls,
        )
        self.test_runs_remaining = task.card.budgets.test_runs
        self.steps: list[AgentStep] = []
        self.transcript: list[dict[str, Any]] = []
        self.terminated = False
        self.finish_reason: FinishReason | None = None
        self.initial_protected: dict[str, str] = {}
        self.policy_violation_count = 0
        self._current_observation: AgentObservation | None = None

    def reset(self) -> AgentObservation:
        """Create a fresh workspace and prove it equals the immutable task snapshot."""

        if self._temporary is not None:
            raise RuntimeError("environment has already been reset")
        self._temporary = tempfile.TemporaryDirectory(prefix=f"nanopt-{self.task.card.id}-")
        root = Path(self._temporary.name) / "workspace"
        copied = copy_snapshot(self.task, root)
        root.chmod(0o755)
        for entry in root.rglob("*"):
            entry.chmod(0o777 if entry.is_dir() else 0o666)
        root.chmod(0o777)
        if copied != self.task.card.snapshot_sha256:
            raise RuntimeError("reset hash differs from task card")
        self.workspace_root = root
        self.workspace = SafeWorkspace(root, self.task.card)
        self.initial_protected = protected_file_hashes(root, self.task.card)
        self.started_at = self.clock()
        self._current_observation = self.observation(None)
        return self._current_observation

    def _require_workspace(self) -> tuple[Path, SafeWorkspace]:
        if self.workspace_root is None or self.workspace is None:
            raise RuntimeError("environment must be reset before use")
        return self.workspace_root, self.workspace

    def budgets(self) -> BudgetState:
        elapsed = max(0.0, self.clock() - self.started_at) if self.started_at else 0.0
        return BudgetState(
            turns_remaining=self.turns_remaining,
            tool_calls_remaining=self.tool_calls_remaining,
            test_runs_remaining=self.test_runs_remaining,
            elapsed_seconds=elapsed,
            wall_clock_limit_seconds=self.task.card.budgets.wall_clock_seconds,
        )

    def observation(self, last_result: ToolResult | None) -> AgentObservation:
        return AgentObservation(
            task_id=self.task.card.id,
            task_version=self.task.card.version,
            issue=self.task.card.issue,
            allowed_tools=self.allowed_tools,
            budgets=self.budgets(),
            last_tool_result=last_result,
            transcript=list(self.transcript),
        )

    def _timed_out(self) -> bool:
        return self.budgets().elapsed_seconds >= self.task.card.budgets.wall_clock_seconds

    @staticmethod
    def _error(code: str, message: str) -> ToolResult:
        return ToolResult(status="error", code=code, message=message)

    def _execute(self, action: Any) -> tuple[ToolResult, list[str]]:
        root, workspace = self._require_workspace()
        violations: list[str] = []
        if action.tool not in self.allowed_tools:
            return self._error("tool_not_allowed", f"tool is not allowed: {action.tool}"), [
                "tool_not_allowed"
            ]
        if action.tool == "list_files":
            result = workspace.list_files(action.arguments.path, action.arguments.max_depth)
        elif action.tool == "read_file":
            result = workspace.read_file(
                action.arguments.path,
                action.arguments.start_line,
                action.arguments.end_line,
            )
        elif action.tool == "search":
            result = workspace.search(
                action.arguments.query,
                action.arguments.path,
                action.arguments.glob,
            )
        elif action.tool == "apply_patch":
            result = workspace.apply_patch(action.arguments.patch)
        elif action.tool == "run_tests":
            if self.test_runs_remaining <= 0:
                return self._error("test_budget_exhausted", "public test budget is exhausted"), []
            self.test_runs_remaining -= 1
            summary = HiddenVerifier(self.backend, self.limits).run_public(self.task, root)
            result = ToolResult(
                status="ok" if summary.status == "passed" else "error",
                code="tests_passed" if summary.status == "passed" else f"tests_{summary.status}",
                message=f"public tests: {summary.passed}/{summary.total}",
                data=summary.model_dump(mode="json"),
                truncated=summary.output_truncated,
            )
            unchanged, changed = protected_files_unchanged(root, self.initial_protected)
            if not unchanged:
                violations.append("protected_files_modified")
                result = self._error(
                    "protected_files_modified",
                    "public tests or submitted code modified protected files: "
                    + ", ".join(changed),
                )
        elif action.tool == "finish":
            if not isinstance(action, FinishAction):
                raise AssertionError("finish action discriminator mismatch")
            result = ToolResult(
                status="ok",
                code="finished",
                message="model requested final verification",
                data={"summary": action.arguments.summary},
            )
            self.terminated = True
            self.finish_reason = "model_finish"
        else:
            raise AssertionError(f"unhandled tool {action.tool}")
        return result, violations

    def step(
        self,
        model_response: str,
        *,
        model_token_ids: list[int] | None = None,
        model_seconds: float | None = None,
    ) -> AgentObservation:
        """Parse and execute one action, charging budgets before exposing the next observation."""

        root, _workspace = self._require_workspace()
        if self.terminated:
            raise RuntimeError("cannot step a terminated environment")
        before = self.budgets()
        if self._timed_out():
            self.terminated = True
            self.finish_reason = "timeout"
            self._current_observation = self.observation(
                self._error("timeout", "episode wall-clock limit reached")
            )
            return self._current_observation
        if self._current_observation is None:
            raise RuntimeError("environment observation state is missing")
        observation = self._current_observation
        self.turns_remaining = max(0, self.turns_remaining - 1)
        self.tool_calls_remaining = max(0, self.tool_calls_remaining - 1)
        started = self.clock()
        action_dict: dict[str, Any] | None = None
        violations: list[str] = []
        try:
            action = parse_action(model_response)
            action_dict = action.model_dump(mode="json")
            parse_status: Literal["valid", "invalid", "error"] = "valid"
            result, violations = self._execute(action)
        except (ValueError, ValidationError) as exc:
            parse_status = "invalid"
            result = self._error("invalid_action", str(exc))
            violations = ["invalid_action"]
        except WorkspacePolicyError as exc:
            parse_status = "valid"
            result = self._error(exc.code, str(exc))
            if exc.code in SECURITY_VIOLATION_CODES:
                violations = [exc.code]
        except Exception as exc:
            parse_status = "error"
            result = self._error("tool_error", f"{type(exc).__name__}: {exc}")
            self.terminated = True
            self.finish_reason = "sandbox_failure"
        self.policy_violation_count += len(violations)
        try:
            state_hash = workspace_sha256(root)
        except WorkspacePolicyError as exc:
            state_hash = None
            violations.append(exc.code)
            self.policy_violation_count += 1
            self.terminated = True
            self.finish_reason = "policy_failure"
        after = self.budgets()
        tool_seconds = max(0.0, self.clock() - started)
        step = AgentStep(
            step_index=len(self.steps),
            observation=observation.model_dump(mode="json"),
            model_response=model_response,
            model_token_ids=model_token_ids,
            action_parse_status=parse_status,
            action=action_dict,
            tool_result=result.model_dump(mode="json"),
            budget_before=before.model_dump(mode="json"),
            budget_after=after.model_dump(mode="json"),
            workspace_state_sha256=state_hash,
            model_seconds=model_seconds,
            tool_seconds=tool_seconds,
            policy_violations=violations,
        )
        self.steps.append(step)
        self.transcript.append(
            {"action": action_dict, "tool_result": result.model_dump(mode="json")}
        )
        if not self.terminated and self._timed_out():
            self.terminated = True
            self.finish_reason = "timeout"
        if not self.terminated and (self.turns_remaining == 0 or self.tool_calls_remaining == 0):
            self.terminated = True
            self.finish_reason = "budget_exhausted"
        self._current_observation = self.observation(result)
        return self._current_observation

    def final_patch(self) -> str:
        """Rebuild an inspectable unified diff from immutable snapshot to final editable files."""

        root, _workspace = self._require_workspace()
        chunks: list[str] = []
        for source in sorted(self.task.snapshot_dir.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(self.task.snapshot_dir)
            final = root / relative
            if not final.is_file() or source.read_bytes() == final.read_bytes():
                continue
            try:
                before = source.read_text(encoding="utf-8").splitlines(keepends=True)
                after = final.read_text(encoding="utf-8").splitlines(keepends=True)
            except UnicodeDecodeError:
                continue
            chunks.extend(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"a/{relative.as_posix()}",
                    tofile=f"b/{relative.as_posix()}",
                )
            )
        return "".join(chunks)

    def run_episode(self, policy: AgentPolicy) -> AgentTrajectory:
        observation = self.reset()
        while not self.terminated:
            try:
                response = policy.respond(observation)
            except Exception as exc:
                self.terminated = True
                self.finish_reason = "policy_failure"
                self.transcript.append({"policy_error": f"{type(exc).__name__}: {exc}"})
                break
            observation = self.step(
                response.text,
                model_token_ids=response.token_ids,
                model_seconds=response.seconds,
            )
        root, _workspace = self._require_workspace()
        verification = HiddenVerifier(self.backend, self.limits).verify(
            self.task,
            root,
            expected_protected_files=self.initial_protected,
            policy_violations=self.policy_violation_count,
        )
        try:
            final_hash = workspace_sha256(root)
        except WorkspacePolicyError:
            final_hash = None
        identity = canonical_json(
            {
                "run_id": self.run_id,
                "task_id": self.task.card.id,
                "task_version": self.task.card.version,
                "policy": policy.identity.model_dump(mode="json"),
                "initial_snapshot_sha256": self.task.card.snapshot_sha256,
            }
        )
        return AgentTrajectory(
            trajectory_id=f"agent_{sha256_bytes(identity)[:24]}",
            run_id=self.run_id,
            task_id=self.task.card.id,
            task_version=self.task.card.version,
            environment_version=ENVIRONMENT_VERSION,
            model_checkpoint_id=policy.identity.checkpoint_id,
            policy=policy.identity,
            initial_snapshot_sha256=self.task.card.snapshot_sha256,
            steps=self.steps,
            finish_reason=self.finish_reason or "error",
            final_workspace_sha256=final_hash,
            verification=verification,
        )

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> MiniSWEEnvironment:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def trajectory_semantics(trajectory: AgentTrajectory) -> dict[str, Any]:
    """Remove measured timing while preserving state/action/result/reward replay evidence."""

    value = trajectory.model_dump(mode="json")

    def normalize(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"model_seconds", "tool_seconds"}:
                    item[key] = None
                elif key in {"elapsed_seconds", "duration_seconds"}:
                    item[key] = 0.0
                elif key == "output" and "workspace_sha256" in item:
                    item[key] = None
                else:
                    normalize(child)
        elif isinstance(item, list):
            for child in item:
                normalize(child)

    normalize(value)
    return value
