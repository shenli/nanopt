"""Train a LoRA policy from frozen, exact-token MiniSWE demonstrations."""

from __future__ import annotations

import html
from pathlib import Path

import torch

from nanopt.agent.sft_data import read_agent_sft_dataset, stored_rendered_example
from nanopt.agent.sft_records import AgentSftSummary
from nanopt.config.models import AgentSftExperiment
from nanopt.config.resolver import ResolutionResult
from nanopt.models.adapters import attach_lora_adapter, parameter_counts, save_lora_adapter
from nanopt.models.loading import load_qwen3_base, qwen_chat_terminator_id
from nanopt.models.renderer import ChatRenderer
from nanopt.runtime.artifacts import append_jsonl, sha256_file, write_json, write_text
from nanopt.runtime.run_context import RunContext, create_run_context
from nanopt.sft.checkpoint import save_sft_checkpoint, sha256_directory
from nanopt.sft.data import CompletionOnlyCollator
from nanopt.sft.records import SftMetricRecord
from nanopt.sft.trainer import (
    SftStepMetrics,
    SftTrainingState,
    build_sft_optimizer,
    evaluate_completion_nll,
    train_sft,
)


def _device(name: str) -> torch.device:
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if selected == "auto":
        selected = "cpu"
    if selected not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(selected)


def _write_report(run_dir: Path, summary: AgentSftSummary) -> None:
    accuracy_row = (
        "| Completion-token accuracy | "
        f"{summary.initial_validation_token_accuracy:.2%} | "
        f"{summary.final_validation_token_accuracy:.2%} |"
    )
    memory_line = (
        "Peak CUDA allocated/reserved: "
        f"{summary.peak_allocated_bytes / 1024**3:.3f} / "
        f"{summary.peak_reserved_bytes / 1024**3:.3f} GiB."
    )
    markdown = f"""# NanoPT Agent SFT report

## What was trained

- Run: `{summary.run_id}`
- Context policy: `{summary.context_policy}`
- Exact-token training examples: {summary.train_examples}
- Task-held-out validation examples: {summary.validation_examples}
- Optimizer steps: {summary.optimizer_steps}
- Held-out task IDs: {", ".join(f"`{item}`" for item in summary.held_out_task_ids)}

## Teacher-forced action metrics

| Metric | Initial | Final |
| --- | ---: | ---: |
| Completion NLL | {summary.initial_validation_nll:.6f} | {summary.final_validation_nll:.6f} |
{accuracy_row}

{memory_line}

The trainer consumed the stored token IDs and action masks directly. It did not decode and
re-tokenize demonstrations. These teacher-forced metrics measure action imitation, not task
success; compare base and adapted policies in the Docker environment for behavioral evidence.
"""
    write_text(run_dir / "report.md", markdown)
    write_text(
        run_dir / "report.html",
        "<!doctype html><html><head><meta charset='utf-8'><title>NanoPT Agent SFT</title>"
        "</head><body><pre>" + html.escape(markdown) + "</pre></body></html>\n",
    )


def execute_agent_sft_run(
    result: ResolutionResult,
    *,
    dataset_dir: Path,
    artifacts_root: Path,
    run_id: str | None,
    local_files_only: bool,
    device: str,
) -> RunContext:
    """Validate lineage, train only LoRA tensors, checkpoint, and report."""

    experiment = result.config.experiment
    if not isinstance(experiment, AgentSftExperiment):
        raise ValueError("Agent SFT execution requires an agent_sft experiment")
    if result.config.model.adapter is None:
        raise ValueError("model profile must define a LoRA adapter")
    manifest, records = read_agent_sft_dataset(dataset_dir)
    if manifest.dataset_id != experiment.data.dataset:
        raise ValueError("dataset ID differs from the Agent SFT profile")
    if manifest.context_policy != experiment.data.context_policy:
        raise ValueError("dataset context policy differs from the Agent SFT profile")
    if manifest.train_tasks != experiment.data.train_tasks:
        raise ValueError("dataset train-task split differs from the Agent SFT profile")
    if manifest.validation_tasks != experiment.data.validation_tasks:
        raise ValueError("dataset validation-task split differs from the Agent SFT profile")

    train_examples = [stored_rendered_example(item) for item in records if item.split == "train"]
    validation_examples = [
        stored_rendered_example(item) for item in records if item.split == "validation"
    ]
    context = create_run_context(result, artifacts_root=artifacts_root, run_id=run_id)
    try:
        context.set_status("running")
        torch.manual_seed(experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(experiment.seed)
        loaded = load_qwen3_base(result.config.model, local_files_only=local_files_only)
        renderer = ChatRenderer(
            loaded.tokenizer,
            enable_thinking=result.config.model.renderer.enable_thinking,
            terminal_token_id=qwen_chat_terminator_id(loaded.tokenizer),
        )
        if renderer.chat_template_sha256 != manifest.chat_template_sha256:
            raise ValueError("runtime chat template differs from the frozen Agent SFT dataset")
        if loaded.tokenizer_revision != manifest.tokenizer_revision:
            raise ValueError("runtime tokenizer revision differs from the Agent SFT dataset")
        if any(item.chat_template_sha256 != renderer.chat_template_sha256 for item in records):
            raise ValueError("an Agent SFT example has a mismatched chat-template hash")

        policy = attach_lora_adapter(
            loaded.model,
            result.config.model.adapter,
            adapter_name=experiment.adapter.name,
        )
        selected_device = _device(device)
        policy.to(selected_device)
        if experiment.training.gradient_checkpointing:
            policy.gradient_checkpointing_enable()
            policy.enable_input_require_grads()
        policy.config.use_cache = False
        counts = parameter_counts(policy)
        context.manifest["model"].update(
            {
                "resolved_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "chat_template_sha256": renderer.chat_template_sha256,
                "base_parameter_count": loaded.parameters.total,
                "trainable_parameter_count": counts.trainable,
                "adapter_name": experiment.adapter.name,
            }
        )
        context.manifest["data"]["fingerprints"].update(
            {
                "agent_sft_dataset": manifest.dataset_sha256,
                "examples": manifest.examples_sha256,
                "source_trajectories": manifest.source_trajectories_sha256,
            }
        )
        context.manifest["data"]["protected_splits_used_for_training"] = False
        context.manifest["training"] = {
            "device": selected_device.type,
            "exact_stored_token_ids": True,
            "context_policy": experiment.data.context_policy,
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        }
        context.set_status("running")

        collator = CompletionOnlyCollator(
            pad_token_id=int(loaded.tokenizer.pad_token_id),
            max_sequence_length=experiment.data.max_sequence_length,
        )
        initial = evaluate_completion_nll(
            policy,
            validation_examples,
            collator,
            micro_batch_size=experiment.training.micro_batch_size,
            device=selected_device,
        )
        append_jsonl(
            context.run_dir / "metrics.jsonl",
            SftMetricRecord(
                run_id=context.manifest["run_id"],
                split="validation",
                optimizer_step=0,
                completion_nll=initial[0],
                completion_token_accuracy=initial[1],
                active_tokens=initial[2],
            ).model_dump(mode="json"),
        )
        optimizer = build_sft_optimizer(policy, experiment.training)
        final_validation = initial
        latest_checkpoint: Path | None = None

        def on_step(
            metric: SftStepMetrics,
            state: SftTrainingState,
            step_optimizer: torch.optim.Optimizer,
        ) -> None:
            nonlocal final_validation, latest_checkpoint
            append_jsonl(
                context.run_dir / "metrics.jsonl",
                SftMetricRecord(
                    run_id=context.manifest["run_id"],
                    split="train",
                    optimizer_step=metric.optimizer_step,
                    completion_nll=metric.completion_nll,
                    completion_token_accuracy=metric.completion_token_accuracy,
                    active_tokens=metric.active_tokens,
                    learning_rate=metric.learning_rate,
                    gradient_norm=metric.gradient_norm,
                    gradient_clipped=metric.gradient_clipped,
                    tokens_per_second=metric.tokens_per_second,
                    peak_allocated_bytes=metric.peak_allocated_bytes,
                    peak_reserved_bytes=metric.peak_reserved_bytes,
                ).model_dump(mode="json"),
            )
            final_step = state.optimizer_step == state.total_optimizer_steps
            if (
                state.optimizer_step % experiment.training.eval_every_optimizer_steps == 0
                or final_step
            ):
                final_validation = evaluate_completion_nll(
                    policy,
                    validation_examples,
                    collator,
                    micro_batch_size=experiment.training.micro_batch_size,
                    device=selected_device,
                )
                append_jsonl(
                    context.run_dir / "metrics.jsonl",
                    SftMetricRecord(
                        run_id=context.manifest["run_id"],
                        split="validation",
                        optimizer_step=state.optimizer_step,
                        completion_nll=final_validation[0],
                        completion_token_accuracy=final_validation[1],
                        active_tokens=final_validation[2],
                    ).model_dump(mode="json"),
                )
            if (
                state.optimizer_step % experiment.training.save_every_optimizer_steps == 0
                or final_step
            ):
                latest_checkpoint = (
                    context.run_dir / "checkpoints" / f"step-{state.optimizer_step:06d}"
                )
                save_sft_checkpoint(
                    policy,
                    step_optimizer,
                    state,
                    latest_checkpoint,
                    adapter_name=experiment.adapter.name,
                )

        training = train_sft(
            policy,
            train_examples,
            collator,
            experiment.training,
            seed=experiment.seed,
            device=selected_device,
            optimizer=optimizer,
            on_step=on_step,
        )
        if latest_checkpoint is None:
            raise RuntimeError("Agent SFT did not produce a checkpoint")
        adapter_dir = save_lora_adapter(
            policy,
            context.run_dir / "adapter",
            adapter_name=experiment.adapter.name,
        )
        adapter_sha = sha256_directory(adapter_dir)
        context.manifest["model"]["adapter_sha256"] = adapter_sha
        context.manifest["checkpoint"] = {
            "path": adapter_dir.relative_to(context.run_dir).as_posix(),
            "sha256": adapter_sha,
            "parent_checkpoint_sha256": None,
        }
        summary = AgentSftSummary(
            run_id=context.manifest["run_id"],
            optimizer_steps=training.state.optimizer_step,
            train_examples=len(train_examples),
            validation_examples=len(validation_examples),
            initial_validation_nll=initial[0],
            final_validation_nll=final_validation[0],
            initial_validation_token_accuracy=initial[1],
            final_validation_token_accuracy=final_validation[1],
            peak_allocated_bytes=max(
                (item.peak_allocated_bytes for item in training.metrics), default=0
            ),
            peak_reserved_bytes=max(
                (item.peak_reserved_bytes for item in training.metrics), default=0
            ),
            context_policy=experiment.data.context_policy,
            held_out_task_ids=experiment.evaluation.held_out_tasks,
        )
        write_json(context.run_dir / "summary.json", summary.model_dump(mode="json"))
        _write_report(context.run_dir, summary)
        context.manifest["artifacts"] = [
            {"path": name, "kind": kind, "sha256": sha256_file(context.run_dir / name)}
            for name, kind in (
                ("metrics.jsonl", "agent_sft_metrics"),
                ("summary.json", "agent_sft_summary"),
                ("report.md", "markdown_report"),
                ("report.html", "html_report"),
            )
        ] + [
            {
                "path": "adapter",
                "kind": "lora_adapter",
                "sha256": adapter_sha,
            }
        ]
        context.set_status("completed")
        return context
    except Exception as exc:
        context.set_status(
            "failed",
            failure={"type": type(exc).__name__, "message": str(exc), "phase": "agent_sft"},
        )
        raise
