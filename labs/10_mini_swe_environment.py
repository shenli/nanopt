"""Run and replay one trusted MiniSWE oracle without a model, Docker, or network."""

from __future__ import annotations

from pathlib import Path

from nanopt.agent.environment import MiniSWEEnvironment, trajectory_semantics
from nanopt.agent.policy import ReplayPolicy, ScriptedOraclePolicy
from nanopt.agent.records import ToolName
from nanopt.agent.sandbox import FakeSandboxBackend, SandboxLimits
from nanopt.agent.tasks import load_task_suite


def main() -> None:
    """Show reset hashes, structured steps, hidden verification, and exact replay."""

    task = load_task_suite(Path("tasks/mini_swe_v1"), split="smoke")[0]
    tools: list[ToolName] = [
        "list_files",
        "read_file",
        "search",
        "apply_patch",
        "run_tests",
        "finish",
    ]
    limits = SandboxLimits(timeout_seconds=10, memory_mib=512, pids=64, cpus=1)
    oracle = ScriptedOraclePolicy(task.oracle_patch_path.read_text(encoding="utf-8"))

    with MiniSWEEnvironment(
        task,
        FakeSandboxBackend(),
        run_id="mini-swe-lab",
        allowed_tools=tools,
        limits=limits,
    ) as environment:
        trajectory = environment.run_episode(oracle)

    replay = ReplayPolicy(
        [step.model_response for step in trajectory.steps],
        trajectory.policy,
    )
    with MiniSWEEnvironment(
        task,
        FakeSandboxBackend(),
        run_id="mini-swe-lab",
        allowed_tools=tools,
        limits=limits,
    ) as environment:
        repeated = environment.run_episode(replay)

    print("Task:             ", trajectory.task_id)
    print("Reset SHA-256:    ", trajectory.initial_snapshot_sha256)
    print("Structured tools: ", [step.action["tool"] for step in trajectory.steps if step.action])
    print("Public tests:     ", trajectory.verification.public.status)
    print("Hidden tests:     ", trajectory.verification.hidden.status)
    print("Hidden output:    ", trajectory.verification.hidden.output)
    print("Final score:      ", trajectory.verification.final_score)

    assert trajectory.initial_snapshot_sha256 == task.card.snapshot_sha256
    assert trajectory.verification.public.status == "passed"
    assert trajectory.verification.hidden.status == "passed"
    assert trajectory.verification.hidden.output is None
    assert trajectory_semantics(trajectory) == trajectory_semantics(repeated)
    print("MiniSWE reset and semantic replay lab passed.")


if __name__ == "__main__":
    main()
