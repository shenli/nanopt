"""Orchestrate fresh grouped rollouts and synchronous GRPO/RLVR updates."""

from __future__ import annotations

import html
import time
from pathlib import Path

import torch

from nanopt.config.models import GrpoExperiment
from nanopt.config.resolver import ResolutionResult
from nanopt.eval.io import (
    read_arithmetic_tasks,
    read_split_manifest,
    validate_tasks_against_manifest,
)
from nanopt.eval.parser import answer_stop_token_ids
from nanopt.grpo.records import GrpoMetricRecord, GrpoSummary, GrpoTrajectoryRecord
from nanopt.grpo.reward import reward_hacking_suite
from nanopt.grpo.rollout import deterministic_prompt_schedule, generate_grouped_trajectory
from nanopt.grpo.trainer import (
    GrpoUpdateMetrics,
    attach_reference_logps,
    build_grpo_optimizer,
    update_grpo_policy,
)
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


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def _iteration_metric(
    run_id: str,
    iteration: int,
    trajectories: list[GrpoTrajectoryRecord],
    update: GrpoUpdateMetrics,
    *,
    rollout_seconds: float,
) -> GrpoMetricRecord:
    completions = [
        completion for trajectory in trajectories for completion in trajectory.completions
    ]
    rewards = torch.tensor([completion.reward for completion in completions], dtype=torch.float32)
    advantages = torch.tensor(
        [completion.advantage for completion in completions], dtype=torch.float32
    )
    lengths = [sum(completion.action_mask) for completion in completions]
    prompt_count = len(trajectories)
    completion_count = len(completions)
    return GrpoMetricRecord(
        run_id=run_id,
        iteration=iteration,
        prompt_count=prompt_count,
        completion_count=completion_count,
        active_tokens=update.active_tokens,
        reward_mean=float(rewards.mean().item()),
        reward_std=float(rewards.std(unbiased=False).item()),
        reward_min=float(rewards.min().item()),
        reward_max=float(rewards.max().item()),
        correctness_rate=sum(completion.verifier_status == "correct" for completion in completions)
        / completion_count,
        parser_success_rate=sum(completion.parser_status == "valid" for completion in completions)
        / completion_count,
        group_reward_std_mean=sum(trajectory.group_reward_std for trajectory in trajectories)
        / prompt_count,
        degenerate_group_fraction=sum(
            trajectory.group_reward_std == 0 for trajectory in trajectories
        )
        / prompt_count,
        advantage_mean=float(advantages.mean().item()),
        advantage_std=float(advantages.std(unbiased=False).item()),
        advantage_max_abs=float(advantages.abs().max().item()),
        completion_length_mean=sum(lengths) / completion_count,
        protocol_stop_fraction=sum(
            completion.finish_reason == "protocol_stop" for completion in completions
        )
        / completion_count,
        eos_fraction=sum(completion.finish_reason == "eos" for completion in completions)
        / completion_count,
        max_length_fraction=sum(
            completion.finish_reason == "max_length" for completion in completions
        )
        / completion_count,
        policy_loss=update.policy_loss,
        kl_loss=update.kl_loss,
        total_loss=update.total_loss,
        clip_fraction=update.clip_fraction,
        ratio_mean=update.ratio_mean,
        ratio_p95=update.ratio_p95,
        current_minus_old_logp_mean=update.current_minus_old_logp_mean,
        sampled_action_surprisal=update.sampled_action_surprisal,
        learning_rate=update.learning_rate,
        gradient_norm=update.gradient_norm,
        gradient_clipped=update.gradient_clipped,
        rollout_seconds=rollout_seconds,
        training_seconds=update.training_seconds,
        peak_allocated_bytes=update.peak_allocated_bytes,
        peak_reserved_bytes=update.peak_reserved_bytes,
    )


def _write_report(run_dir: Path, summary: GrpoSummary) -> None:
    markdown = f"""# NanoPT GRPO/RLVR Report

> On-policy training evidence. Protected capability is established by a separate frozen evaluation.

## Lineage and objective

- Run: `{summary.run_id}`
- Parent DPO adapter: `{summary.parent_dpo_adapter_sha256}`
- GRPO adapter: `{summary.grpo_adapter_sha256}`
- Iterations / optimizer steps: {summary.iterations} / {summary.optimizer_steps}
- Advantage: `{summary.advantage_mode}`
- Loss normalization: `{summary.loss_normalization}`
- Clip epsilon: {summary.clip_epsilon}
- KL: beta={summary.kl_beta}, estimator=`{summary.kl_estimator}`
- Representative run: {str(summary.representative).lower()}

## Fresh rollout aggregate

| Metric | Value |
| --- | ---: |
| Trajectories | {summary.trajectories} |
| Completions | {summary.completions} |
| Mean reward | {summary.mean_reward:.4f} |
| Exact correctness | {summary.correctness_rate:.2%} |
| Parser success | {summary.parser_success_rate:.2%} |
| Degenerate groups | {summary.degenerate_group_fraction:.2%} |
| Mean clip fraction | {summary.mean_clip_fraction:.2%} |

Peak reserved memory was {summary.peak_reserved_bytes / 1024**3:.3f} GiB. Each trajectory stores
sampled IDs, action masks, old behavior log probabilities, finish reason, reward components, and
advantages before optimization. Training pads those IDs directly; decoded reward text is never
re-tokenized.
"""
    write_text(run_dir / "report.md", markdown)
    write_text(
        run_dir / "report.html",
        "<!doctype html><html><head><meta charset='utf-8'><title>NanoPT GRPO Report</title>"
        "</head><body><pre>" + html.escape(markdown) + "</pre></body></html>\n",
    )


def execute_grpo_run(
    result: ResolutionResult,
    *,
    tasks_path: Path,
    dpo_adapter_path: Path,
    artifacts_root: Path,
    run_id: str | None,
    local_files_only: bool,
    device: str,
    iteration_limit: int | None = None,
) -> RunContext:
    """Load DPO, clone GRPO, alternate fresh grouped rollouts and clipped updates, then report."""

    experiment = result.config.experiment
    if not isinstance(experiment, GrpoExperiment):
        raise ValueError("GRPO execution requires a grpo experiment profile")
    if experiment.rollout.group_size < 2:
        raise ValueError("GRPO group size must be at least two")
    if experiment.rollout.temperature != 1.0 or experiment.rollout.top_p != 1.0:
        raise ValueError("reference GRPO requires untruncated temperature-one sampling")
    if experiment.optimization.update_epochs != 1:
        raise ValueError("M6 reference path requires one fresh-data update epoch")
    all_tasks = read_arithmetic_tasks(tasks_path)
    dataset_manifest_path = tasks_path.with_name("dataset_manifest.json")
    dataset_manifest = read_split_manifest(dataset_manifest_path)
    validate_tasks_against_manifest(all_tasks, dataset_manifest)
    prompt_tasks = [task for task in all_tasks if task.split == experiment.data.prompt_pool_split]
    if not prompt_tasks:
        raise ValueError("GRPO prompt pool split is empty")
    iterations = experiment.optimization.iterations
    representative = iteration_limit is None
    if iteration_limit is not None:
        if iteration_limit <= 0:
            raise ValueError("iteration_limit must be positive")
        iterations = min(iterations, iteration_limit)

    context = create_run_context(result, artifacts_root=artifacts_root, run_id=run_id)
    try:
        context.set_status("running")
        torch.manual_seed(experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(experiment.seed)
        loaded = load_qwen3_base(result.config.model, local_files_only=local_files_only)
        policy = load_lora_adapter(
            loaded.model,
            dpo_adapter_path,
            adapter_name=experiment.policy.reference_adapter_stage,
            trainable=False,
        )
        clone_lora_adapter(
            policy,
            source_name=experiment.policy.reference_adapter_stage,
            target_name=experiment.policy.policy_adapter_name,
            trainable=True,
        )
        selected_device = _select_device(device)
        policy.to(selected_device)
        if experiment.optimization.gradient_checkpointing:
            policy.gradient_checkpointing_enable()
            policy.enable_input_require_grads()
        policy.config.use_cache = False
        renderer = ChatRenderer(
            loaded.tokenizer,
            enable_thinking=result.config.model.renderer.enable_thinking,
            terminal_token_id=qwen_chat_terminator_id(loaded.tokenizer),
        )
        eos_token_id = qwen_chat_terminator_id(loaded.tokenizer)
        stop_sequence = answer_stop_token_ids(loaded.tokenizer)
        schedule = deterministic_prompt_schedule(
            prompt_tasks,
            iterations=iterations,
            batch_size=experiment.rollout.prompt_batch_size,
            seed=experiment.seed,
        )
        optimizer = build_grpo_optimizer(policy, experiment)
        run_id_value = str(context.manifest["run_id"])
        attacks = reward_hacking_suite(prompt_tasks[0], experiment.reward.components)
        write_json(context.run_dir / "reward_hacking.json", attacks)
        metrics: list[GrpoMetricRecord] = []
        optimizer_steps = 0
        trajectory_count = 0
        completion_count = 0
        for iteration, task_batch in enumerate(schedule):
            rollout_started = time.perf_counter()
            trajectories = [
                generate_grouped_trajectory(
                    policy,
                    loaded.tokenizer,
                    renderer,
                    task,
                    experiment,
                    run_id=run_id_value,
                    iteration=iteration,
                    eos_token_id=eos_token_id,
                    stop_token_sequence=stop_sequence,
                )
                for task in task_batch
            ]
            rollout_seconds = max(0.0, time.perf_counter() - rollout_started)
            if experiment.optimization.kl_beta > 0:
                with selected_adapter(policy, experiment.policy.reference_adapter_stage):
                    attach_reference_logps(
                        policy,
                        trajectories,
                        pad_token_id=int(loaded.tokenizer.pad_token_id),
                        device=selected_device,
                    )
            for trajectory in trajectories:
                append_jsonl(
                    context.run_dir / "trajectories.jsonl",
                    trajectory.model_dump(mode="json"),
                )
                if experiment.artifacts.save_reward_examples:
                    for completion in trajectory.completions:
                        append_jsonl(
                            context.run_dir / "reward_examples.jsonl",
                            {
                                "trajectory_id": trajectory.trajectory_id,
                                "task_id": trajectory.task_id,
                                "completion_index": completion.completion_index,
                                "decoded_text": completion.decoded_text,
                                "reward": completion.reward,
                                "reward_components": completion.reward_components,
                                "parser_status": completion.parser_status,
                                "verifier_status": completion.verifier_status,
                            },
                        )
            update = update_grpo_policy(
                policy,
                trajectories,
                experiment,
                optimizer,
                iteration=iteration,
                pad_token_id=int(loaded.tokenizer.pad_token_id),
                device=selected_device,
            )
            metric = _iteration_metric(
                run_id_value,
                iteration,
                trajectories,
                update,
                rollout_seconds=rollout_seconds,
            )
            append_jsonl(context.run_dir / "metrics.jsonl", metric.model_dump(mode="json"))
            metrics.append(metric)
            optimizer_steps += update.optimizer_steps
            trajectory_count += len(trajectories)
            completion_count += sum(len(item.completions) for item in trajectories)

        adapter_dir = save_lora_adapter(
            policy,
            context.run_dir / "adapter",
            adapter_name=experiment.policy.policy_adapter_name,
        )
        parent_sha = sha256_directory(dpo_adapter_path)
        adapter_sha = sha256_directory(adapter_dir)
        total_completions = sum(metric.completion_count for metric in metrics)
        total_prompts = sum(metric.prompt_count for metric in metrics)
        peak_metric = max(metrics, key=lambda metric: metric.peak_reserved_bytes)
        summary = GrpoSummary(
            run_id=run_id_value,
            iterations=iterations,
            optimizer_steps=optimizer_steps,
            trajectories=trajectory_count,
            completions=completion_count,
            mean_reward=sum(metric.reward_mean * metric.completion_count for metric in metrics)
            / total_completions,
            correctness_rate=sum(
                metric.correctness_rate * metric.completion_count for metric in metrics
            )
            / total_completions,
            parser_success_rate=sum(
                metric.parser_success_rate * metric.completion_count for metric in metrics
            )
            / total_completions,
            degenerate_group_fraction=sum(
                metric.degenerate_group_fraction * metric.prompt_count for metric in metrics
            )
            / total_prompts,
            mean_clip_fraction=sum(metric.clip_fraction for metric in metrics) / len(metrics),
            parent_dpo_adapter_sha256=parent_sha,
            grpo_adapter_sha256=adapter_sha,
            peak_allocated_bytes=peak_metric.peak_allocated_bytes,
            peak_reserved_bytes=peak_metric.peak_reserved_bytes,
            representative=representative,
            advantage_mode=experiment.advantage.mode,
            loss_normalization=experiment.optimization.loss_normalization,
            clip_epsilon=experiment.optimization.clip_epsilon,
            kl_beta=experiment.optimization.kl_beta,
            kl_estimator=experiment.optimization.kl_estimator,
        )
        write_json(context.run_dir / "summary.json", summary.model_dump(mode="json"))
        _write_report(context.run_dir, summary)
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
                "task_file_sha256": sha256_file(tasks_path),
                "split_manifest_sha256": sha256_file(dataset_manifest_path),
                "dataset": dataset_manifest.dataset_fingerprint,
            }
        )
        context.manifest["training"] = {
            "device": selected_device.type,
            "iterations": iterations,
            "optimizer_steps": optimizer_steps,
            "trajectories": trajectory_count,
            "completions": completion_count,
            "representative": representative,
            "consumed_exact_stored_token_ids": True,
        }
        artifact_specs = [
            ("summary.json", "grpo_summary"),
            ("metrics.jsonl", "grpo_metrics"),
            ("trajectories.jsonl", "rlvr_trajectories"),
            ("reward_examples.jsonl", "reward_examples"),
            ("reward_hacking.json", "reward_hacking_suite"),
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
            if (context.run_dir / name).is_file()
        ]
        context.set_status("completed")
        return context
    except Exception as exc:
        context.set_status(
            "failed",
            failure={"type": type(exc).__name__, "message": str(exc), "phase": "grpo"},
        )
        raise
