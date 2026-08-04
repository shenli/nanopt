from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from nanopt.agent.environment import MiniSWEEnvironment, trajectory_semantics
from nanopt.agent.policy import ReplayPolicy, ScriptedOraclePolicy
from nanopt.agent.sandbox import FakeSandboxBackend, SandboxLimits
from nanopt.agent.tasks import load_task_suite

TOOLS = ["list_files", "read_file", "search", "apply_patch", "run_tests", "finish"]


def test_scripted_oracle_solves_every_task_and_replays_exactly(project_root: Path) -> None:
    schema = json.loads((project_root / "specs/schemas/agent_trajectory.schema.json").read_text())
    for task in load_task_suite(project_root / "tasks/mini_swe_v1", split="smoke"):
        oracle = ScriptedOraclePolicy(task.oracle_patch_path.read_text())
        with MiniSWEEnvironment(
            task,
            FakeSandboxBackend(),
            run_id="oracle",
            allowed_tools=TOOLS,
            limits=SandboxLimits(10, 256, 32),
        ) as environment:
            trajectory = environment.run_episode(oracle)
            assert not (environment.workspace_root / ".nanopt_hidden_tests").exists()
        assert trajectory.verification.public.status == "passed"
        assert trajectory.verification.hidden.status == "passed"
        assert trajectory.verification.hidden.output is None
        assert trajectory.verification.final_score == 1
        assert (
            trajectory.verification.public.workspace_sha256
            == trajectory.verification.hidden.workspace_sha256
        )
        assert trajectory.finish_reason == "model_finish"
        jsonschema.Draft202012Validator(schema).validate(trajectory.model_dump(mode="json"))
        serialized = trajectory.model_dump_json()
        assert ".nanopt_hidden_tests" not in serialized
        assert "HiddenClampTests" not in serialized

        replay = ReplayPolicy([step.model_response for step in trajectory.steps], trajectory.policy)
        with MiniSWEEnvironment(
            task,
            FakeSandboxBackend(),
            run_id="oracle",
            allowed_tools=TOOLS,
            limits=SandboxLimits(10, 256, 32),
        ) as environment:
            repeated = environment.run_episode(replay)
        assert trajectory_semantics(trajectory) == trajectory_semantics(repeated)


def test_invalid_actions_consume_budget_and_terminate(project_root: Path) -> None:
    task = load_task_suite(project_root / "tasks/mini_swe_v1", split="smoke")[0]
    with MiniSWEEnvironment(
        task,
        FakeSandboxBackend(),
        run_id="invalid",
        allowed_tools=TOOLS,
        limits=SandboxLimits(10, 256, 32),
        turn_limit=2,
    ) as environment:
        environment.reset()
        environment.step('{"tool":"shell","arguments":{"command":"id"}}')
        environment.step("not-json")
        assert environment.terminated
        assert environment.finish_reason == "budget_exhausted"
        assert environment.policy_violation_count == 2


def test_explicit_tool_call_limit_caps_an_identical_task_snapshot(project_root: Path) -> None:
    task = load_task_suite(project_root / "tasks/mini_swe_v1", split="smoke")[0]
    with MiniSWEEnvironment(
        task,
        FakeSandboxBackend(),
        run_id="tool-budget",
        allowed_tools=TOOLS,
        limits=SandboxLimits(10, 256, 32),
        tool_call_limit=2,
    ) as environment:
        trajectory = environment.run_episode(
            ScriptedOraclePolicy(task.oracle_patch_path.read_text())
        )

    assert len(trajectory.steps) == 2
    assert trajectory.finish_reason == "budget_exhausted"
    assert trajectory.verification.final_score < 1.0
