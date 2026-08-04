"""Orchestrate fresh MiniSWE rollout groups and exact-token Agent GRPO updates."""

from __future__ import annotations

import html
import random
import time
from pathlib import Path
from typing import Any

import torch

from nanopt.agent.rl_records import (
    AgentRlBudgetPoint,
    AgentRlBudgetStudy,
    AgentRlMetric,
    AgentRlStalenessStudy,
    AgentRlSummary,
)
from nanopt.agent.rl_rollout import generate_agent_rl_episode, generate_agent_rl_group
from nanopt.agent.rl_trainer import (
    AgentRlUpdateMetrics,
    attach_agent_rl_reference_logps,
    build_agent_rl_optimizer,
    build_credit_assignment_study,
    flatten_agent_rl_actions,
    measure_agent_rl_staleness,
    update_agent_rl_policy,
)
from nanopt.agent.sandbox import (
    DockerSandboxBackend,
    FakeSandboxBackend,
    SandboxBackend,
    SandboxLimits,
)
from nanopt.agent.tasks import LoadedAgentTask, load_task_suite
from nanopt.config.models import AgentRlExperiment
from nanopt.config.resolver import ResolutionResult
from nanopt.models.adapters import (
    clone_lora_adapter,
    load_lora_adapter,
    parameter_counts,
    save_lora_adapter,
    selected_adapter,
)
from nanopt.models.loading import load_qwen3_base, qwen_chat_terminator_id
from nanopt.models.renderer import ChatRenderer
from nanopt.runtime.artifacts import append_jsonl, sha256_file, write_json, write_text
from nanopt.runtime.run_context import RunContext, create_run_context
from nanopt.sft.checkpoint import sha256_directory


def _device(name: str) -> torch.device:
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if selected == "auto":
        selected = "cpu"
    if selected not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(selected)


def _limits(experiment: AgentRlExperiment) -> SandboxLimits:
    return SandboxLimits(
        timeout_seconds=min(60, experiment.environment.wall_clock_timeout_seconds),
        memory_mib=experiment.environment.memory_limit_mib,
        pids=experiment.environment.pids_limit,
        cpus=experiment.environment.cpu_limit,
    )


def _backend(experiment: AgentRlExperiment) -> tuple[SandboxBackend, dict[str, object] | None]:
    if experiment.environment.backend == "docker":
        docker = DockerSandboxBackend(experiment.environment.image)
        return docker, docker.validate_available()
    return FakeSandboxBackend(), None


def _task_schedule(
    tasks: list[LoadedAgentTask], *, iterations: int, seed: int
) -> list[LoadedAgentTask]:
    """Repeat shuffled task epochs without consuming model sampling RNG state."""

    if not tasks or iterations <= 0:
        raise ValueError("Agent RL task schedule needs tasks and positive iterations")
    rng = random.Random(seed)
    result: list[LoadedAgentTask] = []
    while len(result) < iterations:
        epoch = sorted(tasks, key=lambda task: task.card.id)
        rng.shuffle(epoch)
        result.extend(epoch)
    return result[:iterations]


def _budget_point(
    model: Any,
    tokenizer: Any,
    renderer: ChatRenderer,
    tasks: list[LoadedAgentTask],
    experiment: AgentRlExperiment,
    backend: SandboxBackend,
    limits: SandboxLimits,
    *,
    run_id: str,
    checkpoint: str,
    policy_version: int,
    tool_budget: int,
) -> AgentRlBudgetPoint:
    episodes = [
        generate_agent_rl_episode(
            model,
            tokenizer,
            renderer,
            task,
            experiment,
            backend,
            limits,
            run_id=f"{run_id}-budget-{checkpoint}-{tool_budget}",
            iteration=experiment.rollout.iterations + tool_budget,
            rollout_index=index,
            policy_version=policy_version,
            checkpoint_id=checkpoint,
            do_sample=False,
            tool_call_limit=tool_budget,
        )
        for index, task in enumerate(tasks)
    ]
    actions = [action for episode in episodes for action in episode.actions]
    valid = sum(action.action_parse_status == "valid" for action in actions)
    return AgentRlBudgetPoint(
        checkpoint=checkpoint,  # type: ignore[arg-type]
        tool_budget=tool_budget,
        tasks=len(episodes),
        solved=sum(episode.hidden_outcome_reward == 1.0 for episode in episodes),
        mean_hidden_outcome_reward=sum(episode.hidden_outcome_reward for episode in episodes)
        / len(episodes),
        action_validity_rate=valid / len(actions),
    )


def _metric(
    run_id: str,
    iteration: int,
    group: Any,
    update: AgentRlUpdateMetrics,
    *,
    rollout_seconds: float,
) -> AgentRlMetric:
    episodes = group.episodes
    actions = [action for episode in episodes for action in episode.actions]
    valid = sum(action.action_parse_status == "valid" for action in actions)
    return AgentRlMetric(
        run_id=run_id,
        iteration=iteration,
        policy_version_before=group.policy_version,
        policy_version_after=group.policy_version + 1,
        episodes=len(episodes),
        actions=len(actions),
        active_tokens=update.active_tokens,
        reward_mean=group.reward_mean,
        reward_std=group.reward_std,
        solved_rate=sum(episode.hidden_outcome_reward == 1.0 for episode in episodes)
        / len(episodes),
        action_validity_rate=valid / len(actions),
        degenerate_group=group.degenerate,
        policy_loss=update.policy_loss,
        kl_loss=update.kl_loss,
        total_loss=update.total_loss,
        clip_fraction=update.clip_fraction,
        ratio_mean=update.ratio_mean,
        gradient_norm=update.gradient_norm,
        optimizer_steps=update.optimizer_steps,
        rollout_seconds=rollout_seconds,
        training_seconds=update.training_seconds,
        peak_allocated_bytes=update.peak_allocated_bytes,
        peak_reserved_bytes=update.peak_reserved_bytes,
    )


def _write_report(
    run_dir: Path,
    summary: AgentRlSummary,
    staleness: AgentRlStalenessStudy,
    budget: AgentRlBudgetStudy,
) -> None:
    staleness_rows = "\n".join(
        f"| {point.label.title()} | {point.policy_lag} | {point.mean_abs_log_ratio:.6f} | "
        f"{point.max_abs_log_ratio:.6f} | {point.approximate_ess_fraction:.3f} |"
        for point in (staleness.fresh, staleness.stale)
    )
    budget_rows = "\n".join(
        f"| `{point.checkpoint}` | {point.tool_budget} | {point.solved}/{point.tasks} | "
        f"{point.mean_hidden_outcome_reward:.3f} | {point.action_validity_rate:.1%} |"
        for point in budget.points
    )
    markdown = f"""# NanoPT Mini Agent RL report

## Training contract

- Run: `{summary.run_id}`
- Parent Agent SFT adapter: `{summary.parent_agent_sft_adapter_sha256}`
- Agent RL adapter: `{summary.agent_rl_adapter_sha256}`
- Fresh iterations / optimizer steps: {summary.iterations} / {summary.optimizer_steps}
- Groups / episodes / action turns: {summary.groups} / {summary.episodes} / {summary.actions}
- Maximum training policy lag: {summary.maximum_training_policy_lag}
- Hidden reward exposed during rollout: {str(summary.hidden_reward_exposed_during_rollout).lower()}
- Exact sampled token IDs consumed: {str(summary.exact_sampled_tokens).lower()}

Each group used independently reset copies of one immutable task snapshot. The hidden verifier ran
only after an episode terminated. Every optimizer input came from the current policy version; the
retained stale group below was measured but never reused for training.

## Outcome

| Metric | Value |
| --- | ---: |
| Mean sampled training reward | {summary.mean_reward:.3f} |
| Sampled action validity | {summary.action_validity_rate:.1%} |
| Degenerate rollout groups | {summary.degenerate_group_fraction:.1%} |
| Held-out greedy reward before RL | {summary.initial_validation_reward:.3f} |
| Held-out greedy reward after RL | {summary.final_validation_reward:.3f} |
| Peak reserved CUDA memory | {summary.peak_reserved_bytes / 1024**3:.3f} GiB |

## Fresh versus stale trajectories

| Point | Policy lag | Mean |log ratio| | Max |log ratio| | Approximate ESS |
| --- | ---: | ---: | ---: | ---: |
{staleness_rows}

## Tool-budget study

| Checkpoint | Tool budget | Solved | Mean hidden reward | Action validity |
| --- | ---: | ---: | ---: | ---: |
{budget_rows}

These measurements cover a five-task educational suite and do not establish general coding-agent
performance. Terminal-only credit assignment is retained as a counterfactual coverage study; the
reference optimizer assigns the group-relative outcome advantage to every sampled action token.
"""
    write_text(run_dir / "report.md", markdown)
    write_text(
        run_dir / "report.html",
        "<!doctype html><html><head><meta charset='utf-8'><title>NanoPT Agent RL report"
        "</title></head><body><pre>" + html.escape(markdown) + "</pre></body></html>\n",
    )


def execute_agent_rl_run(
    result: ResolutionResult,
    *,
    tasks_root: Path,
    agent_sft_adapter_path: Path,
    artifacts_root: Path,
    run_id: str | None,
    local_files_only: bool,
    device: str,
    iteration_limit: int | None = None,
) -> RunContext:
    """Train Agent RL from fresh exact-token groups and retain all required v0.3 studies."""

    experiment = result.config.experiment
    if not isinstance(experiment, AgentRlExperiment):
        raise ValueError("Agent RL execution requires an agent_rl experiment profile")
    iterations = experiment.rollout.iterations
    representative = iteration_limit is None
    if iteration_limit is not None:
        if iteration_limit <= 0:
            raise ValueError("iteration_limit must be positive")
        iterations = min(iterations, iteration_limit)
    all_tasks = load_task_suite(tasks_root, split="all")
    by_id = {task.card.id: task for task in all_tasks}
    requested = set(experiment.tasks.train_tasks + experiment.tasks.validation_tasks)
    missing = sorted(requested - set(by_id))
    if missing:
        raise ValueError(f"Agent RL profile names unknown tasks: {', '.join(missing)}")
    train_tasks = [by_id[task_id] for task_id in experiment.tasks.train_tasks]
    validation_tasks = [by_id[task_id] for task_id in experiment.tasks.validation_tasks]
    schedule = _task_schedule(train_tasks, iterations=iterations, seed=experiment.seed)
    selected_device = _device(device)
    backend, docker_evidence = _backend(experiment)
    limits = _limits(experiment)
    context = create_run_context(result, artifacts_root=artifacts_root, run_id=run_id)

    try:
        context.set_status("running")
        torch.manual_seed(experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(experiment.seed)
            torch.cuda.reset_peak_memory_stats(selected_device)
        loaded = load_qwen3_base(result.config.model, local_files_only=local_files_only)
        policy = load_lora_adapter(
            loaded.model,
            agent_sft_adapter_path,
            adapter_name=experiment.policy.reference_adapter_name,
            trainable=False,
        )
        clone_lora_adapter(
            policy,
            source_name=experiment.policy.reference_adapter_name,
            target_name=experiment.policy.policy_adapter_name,
            trainable=True,
        )
        policy.to(selected_device)
        if experiment.optimization.gradient_checkpointing:
            policy.gradient_checkpointing_enable()
            policy.enable_input_require_grads()
        policy.config.use_cache = False
        tokenizer = loaded.tokenizer
        pad_token_id = int(tokenizer.pad_token_id)
        renderer = ChatRenderer(
            tokenizer,
            enable_thinking=result.config.model.renderer.enable_thinking,
            terminal_token_id=qwen_chat_terminator_id(tokenizer),
        )
        optimizer = build_agent_rl_optimizer(policy, experiment)
        run_id_value = str(context.manifest["run_id"])

        reference_points: list[AgentRlBudgetPoint] = []
        with selected_adapter(policy, experiment.policy.reference_adapter_name):
            for budget in experiment.studies.tool_budgets:
                reference_points.append(
                    _budget_point(
                        policy,
                        tokenizer,
                        renderer,
                        validation_tasks,
                        experiment,
                        backend,
                        limits,
                        run_id=run_id_value,
                        checkpoint="reference",
                        policy_version=0,
                        tool_budget=budget,
                    )
                )

        groups: list[Any] = []
        metrics: list[AgentRlMetric] = []
        optimizer_steps = 0
        policy_version = 0
        for iteration, task in enumerate(schedule):
            rollout_started = time.perf_counter()
            group = generate_agent_rl_group(
                policy,
                tokenizer,
                renderer,
                task,
                experiment,
                backend,
                limits,
                run_id=run_id_value,
                iteration=iteration,
                policy_version=policy_version,
                checkpoint_id=experiment.policy.policy_adapter_name,
            )
            rollout_seconds = max(0.0, time.perf_counter() - rollout_started)
            with selected_adapter(policy, experiment.policy.reference_adapter_name):
                attach_agent_rl_reference_logps(
                    policy,
                    [group],
                    pad_token_id=pad_token_id,
                    device=selected_device,
                )
            append_jsonl(context.run_dir / "rollout_groups.jsonl", group.model_dump(mode="json"))
            update = update_agent_rl_policy(
                policy,
                [group],
                experiment,
                optimizer,
                iteration=iteration,
                policy_version=policy_version,
                pad_token_id=pad_token_id,
                device=selected_device,
            )
            metric = _metric(
                run_id_value,
                iteration,
                group,
                update,
                rollout_seconds=rollout_seconds,
            )
            append_jsonl(context.run_dir / "metrics.jsonl", metric.model_dump(mode="json"))
            groups.append(group)
            metrics.append(metric)
            optimizer_steps += update.optimizer_steps
            policy_version += 1

        # The final policy scores both the newest and oldest retained data. Neither point is fed
        # back into the optimizer, keeping the training policy-lag invariant at exactly zero.
        fresh = measure_agent_rl_staleness(
            policy,
            [groups[-1]],
            label="fresh",
            scored_policy_version=policy_version,
            pad_token_id=pad_token_id,
            device=selected_device,
        )
        stale = measure_agent_rl_staleness(
            policy,
            [groups[0]],
            label="stale",
            scored_policy_version=policy_version,
            pad_token_id=pad_token_id,
            device=selected_device,
        )
        staleness = AgentRlStalenessStudy(
            final_policy_version=policy_version,
            fresh=fresh,
            stale=stale,
        )
        credit = build_credit_assignment_study(groups)

        policy_points = [
            _budget_point(
                policy,
                tokenizer,
                renderer,
                validation_tasks,
                experiment,
                backend,
                limits,
                run_id=run_id_value,
                checkpoint="agent_rl",
                policy_version=policy_version,
                tool_budget=budget,
            )
            for budget in experiment.studies.tool_budgets
        ]
        budget_study = AgentRlBudgetStudy(
            task_ids=experiment.tasks.validation_tasks,
            points=[*reference_points, *policy_points],
        )
        write_json(context.run_dir / "staleness_study.json", staleness.model_dump(mode="json"))
        write_json(context.run_dir / "credit_study.json", credit.model_dump(mode="json"))
        write_json(context.run_dir / "tool_budget_study.json", budget_study.model_dump(mode="json"))

        adapter_dir = save_lora_adapter(
            policy,
            context.run_dir / "adapter",
            adapter_name=experiment.policy.policy_adapter_name,
        )
        parent_sha = sha256_directory(agent_sft_adapter_path)
        adapter_sha = sha256_directory(adapter_dir)
        episodes = [episode for group in groups for episode in group.episodes]
        actions = flatten_agent_rl_actions(groups)
        maximum_budget = max(experiment.studies.tool_budgets)
        initial_validation = next(
            point.mean_hidden_outcome_reward
            for point in reference_points
            if point.tool_budget == maximum_budget
        )
        final_validation = next(
            point.mean_hidden_outcome_reward
            for point in policy_points
            if point.tool_budget == maximum_budget
        )
        summary = AgentRlSummary(
            run_id=run_id_value,
            iterations=iterations,
            optimizer_steps=optimizer_steps,
            groups=len(groups),
            episodes=len(episodes),
            actions=len(actions),
            mean_reward=sum(episode.hidden_outcome_reward for episode in episodes) / len(episodes),
            action_validity_rate=sum(action.action_parse_status == "valid" for action in actions)
            / len(actions),
            degenerate_group_fraction=sum(group.degenerate for group in groups) / len(groups),
            initial_validation_reward=initial_validation,
            final_validation_reward=final_validation,
            parent_agent_sft_adapter_sha256=parent_sha,
            agent_rl_adapter_sha256=adapter_sha,
            peak_allocated_bytes=max(metric.peak_allocated_bytes for metric in metrics),
            peak_reserved_bytes=max(metric.peak_reserved_bytes for metric in metrics),
            representative=representative,
        )
        write_json(context.run_dir / "summary.json", summary.model_dump(mode="json"))
        _write_report(context.run_dir, summary, staleness, budget_study)

        counts = parameter_counts(policy)
        context.manifest["model"].update(
            {
                "resolved_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "chat_template_sha256": renderer.chat_template_sha256,
                "trainable_parameter_count": counts.trainable,
                "parent_adapter_sha256": parent_sha,
                "adapter_name": experiment.policy.policy_adapter_name,
                "adapter_sha256": adapter_sha,
            }
        )
        context.manifest["data"]["fingerprints"].update(
            {
                "task_suite": experiment.tasks.suite,
                "train_tasks": ",".join(experiment.tasks.train_tasks),
                "validation_tasks": ",".join(experiment.tasks.validation_tasks),
            }
        )
        context.manifest["training"] = {
            "device": selected_device.type,
            "iterations": iterations,
            "optimizer_steps": optimizer_steps,
            "groups": len(groups),
            "episodes": len(episodes),
            "actions": len(actions),
            "representative": representative,
            "consumed_exact_stored_token_ids": True,
            "maximum_policy_lag": 0,
            "hidden_reward_exposed_during_rollout": False,
        }
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
            "policy": "model",
            "representative": representative,
            "environment_trains_model": True,
            "task_count": len(train_tasks) + len(validation_tasks),
            "solved": sum(episode.hidden_outcome_reward == 1.0 for episode in episodes),
            "replay_checked": False,
        }
        artifact_specs = [
            ("summary.json", "agent_rl_summary"),
            ("metrics.jsonl", "agent_rl_metrics"),
            ("rollout_groups.jsonl", "agent_rl_exact_rollout_groups"),
            ("staleness_study.json", "agent_rl_staleness_study"),
            ("credit_study.json", "agent_rl_credit_study"),
            ("tool_budget_study.json", "agent_rl_tool_budget_study"),
            ("report.md", "markdown_report"),
            ("report.html", "html_report"),
        ]
        context.manifest["artifacts"] = [
            {
                "path": name,
                "kind": kind,
                "sha256": sha256_file(context.run_dir / name),
            }
            for name, kind in artifact_specs
        ]
        context.set_status("completed")
        return context
    except Exception as exc:
        context.set_status(
            "failed",
            failure={"type": type(exc).__name__, "message": str(exc), "phase": "agent_rl"},
        )
        raise
