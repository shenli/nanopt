"""Run and report one inspectable MiniSWE agent-evaluation suite."""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Literal

from nanopt.agent.environment import MiniSWEEnvironment, trajectory_semantics
from nanopt.agent.policy import QwenStructuredPolicy, ReplayPolicy, ScriptedOraclePolicy
from nanopt.agent.records import AgentRunSummary, AgentTrajectory
from nanopt.agent.sandbox import (
    DockerSandboxBackend,
    FakeSandboxBackend,
    SandboxBackend,
    SandboxLimits,
)
from nanopt.agent.tasks import LoadedAgentTask, load_task_suite
from nanopt.config.models import AgentEvaluationExperiment
from nanopt.config.resolver import ResolutionResult
from nanopt.runtime.artifacts import (
    append_jsonl,
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_text,
)
from nanopt.runtime.run_context import RunContext, create_run_context
from nanopt.sft.checkpoint import sha256_directory


def _suite_fingerprint(tasks: list[LoadedAgentTask]) -> str:
    value = [
        {
            "card": task.card.model_dump(mode="json"),
            "hidden_tests_sha256": sha256_directory(task.hidden_tests_dir),
            "oracle_patch_sha256": sha256_file(task.oracle_patch_path),
        }
        for task in tasks
    ]
    return sha256_bytes(canonical_json(value))


def _write_report(
    run_dir: Path, summary: AgentRunSummary, trajectories: list[AgentTrajectory]
) -> None:
    rows = "\n".join(
        f"| `{item.task_id}` | {item.finish_reason} | "
        f"{item.verification.public.passed}/{item.verification.public.total} | "
        f"{item.verification.hidden.passed}/{item.verification.hidden.total} | "
        f"{item.verification.final_score:.3f} | {len(item.steps)} |"
        for item in trajectories
    )
    markdown = f"""# NanoPT MiniSWE agent-evaluation report

> Evaluation only. This environment does not train or update the language model.

## Run contract

- Run: `{summary.run_id}`
- Backend: `{summary.backend}`
- Policy: `{summary.policy}`
- Tasks solved: {summary.solved}/{summary.tasks}
- Mean hidden-verifier score: {summary.mean_score:.3f}
- Policy violations: {summary.policy_violations}
- Total structured steps: {summary.total_steps}
- Wall time: {summary.wall_seconds:.3f} seconds
- Representative: {str(summary.representative).lower()}

| Task | Finish | Public | Hidden | Score | Steps |
| --- | --- | ---: | ---: | ---: | ---: |
{rows}

Hidden verifier source and output are deliberately absent. Each task was reset from its immutable
snapshot; the Docker reference backend executes trusted test commands as non-root with no network,
GPU, Linux capabilities, or writable root filesystem. Containers reduce risk but are not a perfect
boundary against hostile kernel-level attacks.
"""
    write_text(run_dir / "report.md", markdown)
    write_text(
        run_dir / "report.html",
        "<!doctype html><html><head><meta charset='utf-8'><title>NanoPT MiniSWE report"
        "</title></head><body><pre>" + html.escape(markdown) + "</pre></body></html>\n",
    )


def execute_agent_run(
    result: ResolutionResult,
    *,
    tasks_root: Path,
    policy_kind: Literal["oracle", "model"],
    artifacts_root: Path,
    run_id: str | None,
    adapter_path: Path | None,
    adapter_name: str,
    local_files_only: bool,
    device: str,
    max_tasks: int | None = None,
    turn_limit: int | None = None,
) -> RunContext:
    """Evaluate one policy without training, preserving actions before aggregate scores."""

    experiment = result.config.experiment
    if not isinstance(experiment, AgentEvaluationExperiment):
        raise ValueError("agent execution requires an agent_evaluation experiment")
    tasks = load_task_suite(tasks_root, split=experiment.tasks.split)
    representative = max_tasks is None and turn_limit is None
    if max_tasks is not None:
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError("agent task selection is empty")
    for task in tasks:
        if task.card.budgets.tool_calls > experiment.environment.tool_budget:
            raise ValueError(f"task {task.card.id} exceeds profile tool budget")
        if task.card.budgets.test_runs > experiment.environment.test_run_budget:
            raise ValueError(f"task {task.card.id} exceeds profile test budget")
        if task.card.budgets.turns > experiment.policy.max_turns:
            raise ValueError(f"task {task.card.id} exceeds profile turn budget")

    limits = SandboxLimits(
        timeout_seconds=min(60, experiment.environment.wall_clock_timeout_seconds),
        memory_mib=experiment.environment.memory_limit_mib,
        pids=experiment.environment.pids_limit,
        cpus=experiment.environment.cpu_limit,
    )
    docker_evidence: dict[str, object] | None = None
    if experiment.environment.backend == "docker":
        docker = DockerSandboxBackend(experiment.environment.image)
        docker_evidence = docker.validate_available()
        backend: SandboxBackend = docker
    else:
        backend = FakeSandboxBackend()

    model_policy: QwenStructuredPolicy | None = None
    if policy_kind == "model":
        model_policy = QwenStructuredPolicy(
            result.config.model,
            experiment,
            adapter_path=adapter_path,
            adapter_name=adapter_name,
            local_files_only=local_files_only,
            device=device,
        )

    context = create_run_context(result, artifacts_root=artifacts_root, run_id=run_id)
    trajectories: list[AgentTrajectory] = []
    replay: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    try:
        context.set_status("running")
        trajectory_dir = context.run_dir / "agent_trajectories"
        patch_dir = context.run_dir / "final_patches"
        trajectory_dir.mkdir()
        patch_dir.mkdir()
        for task in tasks:
            policy = (
                ScriptedOraclePolicy(task.oracle_patch_path.read_text(encoding="utf-8"))
                if policy_kind == "oracle"
                else model_policy
            )
            if policy is None:
                raise RuntimeError("model policy was not initialized")
            with MiniSWEEnvironment(
                task,
                backend,
                run_id=context.manifest["run_id"],
                allowed_tools=list(experiment.tools),
                limits=limits,
                turn_limit=turn_limit,
            ) as environment:
                trajectory = environment.run_episode(policy)
                final_patch = environment.final_patch()
            write_json(
                trajectory_dir / f"{task.card.id}.json",
                trajectory.model_dump(mode="json"),
            )
            write_text(patch_dir / f"{task.card.id}.patch", final_patch)
            append_jsonl(
                context.run_dir / "trajectories.jsonl",
                trajectory.model_dump(mode="json"),
            )
            trajectories.append(trajectory)

            if policy_kind == "oracle":
                replay_policy = ReplayPolicy(
                    [step.model_response for step in trajectory.steps], trajectory.policy
                )
                with MiniSWEEnvironment(
                    task,
                    backend,
                    run_id=context.manifest["run_id"],
                    allowed_tools=list(experiment.tools),
                    limits=limits,
                    turn_limit=turn_limit,
                ) as replay_environment:
                    repeated = replay_environment.run_episode(replay_policy)
                exact = trajectory_semantics(trajectory) == trajectory_semantics(repeated)
                replay[task.card.id] = {
                    "exact_semantic_match": exact,
                    "trajectory_sha256": sha256_file(trajectory_dir / f"{task.card.id}.json"),
                }
                if not exact:
                    raise RuntimeError(f"trajectory replay differs for task {task.card.id}")

        wall_seconds = time.perf_counter() - started
        solved = sum(item.verification.final_score == 1.0 for item in trajectories)
        summary = AgentRunSummary(
            run_id=context.manifest["run_id"],
            backend=backend.name,
            policy=policy_kind,
            tasks=len(trajectories),
            solved=solved,
            mean_score=sum(item.verification.final_score for item in trajectories)
            / len(trajectories),
            policy_violations=sum(
                len(step.policy_violations) for item in trajectories for step in item.steps
            ),
            total_steps=sum(len(item.steps) for item in trajectories),
            wall_seconds=wall_seconds,
            representative=representative,
        )
        write_json(context.run_dir / "summary.json", summary.model_dump(mode="json"))
        write_json(context.run_dir / "replay.json", replay)
        _write_report(context.run_dir, summary, trajectories)
        suite_fingerprint = _suite_fingerprint(tasks)
        context.manifest["data"]["fingerprints"].update(
            {
                "agent_task_suite": suite_fingerprint,
                "task_count": str(len(tasks)),
            }
        )
        context.manifest["agent_environment"] = {
            "backend": backend.name,
            "image": experiment.environment.image if backend.name == "docker" else None,
            "docker_evidence": docker_evidence,
            "network": "none" if backend.name == "docker" else "host-test-only",
            "run_as_non_root": backend.name == "docker",
            "expose_gpu": False,
            "capabilities_dropped": backend.name == "docker",
            "no_new_privileges": backend.name == "docker",
            "root_filesystem_read_only": backend.name == "docker",
            "separate_hidden_workspace": True,
            "hidden_source_exposed": False,
            "policy": policy_kind,
            "representative": representative,
            "environment_trains_model": False,
            "task_count": len(tasks),
            "solved": solved,
            "replay_checked": policy_kind == "oracle",
        }
        if model_policy is not None:
            context.manifest["model"].update(
                {
                    "resolved_revision": model_policy.identity.generation["model_revision"],
                    "tokenizer_revision": model_policy.identity.generation["tokenizer_revision"],
                    "adapter_name": adapter_name if adapter_path else None,
                    "adapter_sha256": (
                        model_policy.identity.generation["adapter_sha256"] if adapter_path else None
                    ),
                }
            )
        artifact_names = (
            "summary.json",
            "trajectories.jsonl",
            "replay.json",
            "report.md",
            "report.html",
        )
        context.manifest["artifacts"] = [
            {
                "path": name,
                "kind": f"agent_{Path(name).stem}",
                "sha256": sha256_file(context.run_dir / name),
            }
            for name in artifact_names
        ] + [
            {
                "path": directory,
                "kind": kind,
                "sha256": sha256_directory(context.run_dir / directory),
            }
            for directory, kind in (
                ("agent_trajectories", "agent_trajectory_records"),
                ("final_patches", "agent_final_patches"),
            )
        ]
        context.set_status("completed")
        return context
    except Exception as exc:
        context.set_status(
            "failed",
            failure={"type": type(exc).__name__, "message": str(exc), "phase": "agent_evaluation"},
        )
        raise
