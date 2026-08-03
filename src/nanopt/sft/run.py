"""Orchestrate one inspectable SFT run without hiding the training loop."""

from __future__ import annotations

import html
from pathlib import Path

import torch

from nanopt.config.models import SftExperiment
from nanopt.config.resolver import ResolutionResult
from nanopt.data.schemas import ArithmeticTask
from nanopt.eval.io import (
    read_arithmetic_tasks,
    read_split_manifest,
    validate_tasks_against_manifest,
)
from nanopt.models.adapters import (
    attach_lora_adapter,
    load_lora_adapter,
    parameter_counts,
    save_lora_adapter,
)
from nanopt.models.loading import load_qwen3_base, qwen_chat_terminator_id
from nanopt.models.renderer import ChatRenderer
from nanopt.runtime.artifacts import append_jsonl, sha256_file, write_json, write_text
from nanopt.runtime.run_context import RunContext, create_run_context
from nanopt.sft.checkpoint import (
    read_sft_checkpoint,
    restore_sft_optimizer,
    save_sft_checkpoint,
    sha256_directory,
)
from nanopt.sft.data import CompletionOnlyCollator, render_sft_examples
from nanopt.sft.records import SftMetricRecord, SftSummary
from nanopt.sft.schedule import optimizer_groups
from nanopt.sft.trainer import (
    SftStepMetrics,
    SftTrainingState,
    build_sft_optimizer,
    evaluate_completion_nll,
    train_sft,
)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def _split_tasks(
    tasks: list[ArithmeticTask], experiment: SftExperiment
) -> tuple[list[ArithmeticTask], list[ArithmeticTask]]:
    train = [task for task in tasks if task.split == experiment.data.train_split]
    validation = [task for task in tasks if task.split == experiment.data.validation_split]
    if not train or not validation:
        raise ValueError(
            "SFT task file must contain non-empty configured train and validation splits"
        )
    return train, validation


def _write_report(run_dir: Path, summary: SftSummary) -> None:
    delta = summary.initial_validation_nll - summary.final_validation_nll
    nll_row = (
        f"| Validation completion NLL | {summary.initial_validation_nll:.6f} | "
        f"{summary.final_validation_nll:.6f} |"
    )
    accuracy_row = (
        "| Validation completion-token accuracy | "
        f"{summary.initial_validation_token_accuracy:.2%} | "
        f"{summary.final_validation_token_accuracy:.2%} |"
    )
    markdown = f"""# NanoPT SFT Report

> SFT evidence only. Generation quality must be established by a separate protected evaluation.

## Identity

- Run: `{summary.run_id}`
- Optimizer steps: {summary.optimizer_steps}
- Training examples: {summary.train_examples}
- Validation examples: {summary.validation_examples}
- Representative training run: {str(summary.representative).lower()}

## Teacher-forced metrics

| Metric | Initial | Final |
| --- | ---: | ---: |
{nll_row}
{accuracy_row}

Validation NLL improvement: {delta:.6f}.

## Memory

- Peak allocated: {summary.peak_allocated_bytes / 1024**3:.3f} GiB
- Peak reserved: {summary.peak_reserved_bytes / 1024**3:.3f} GiB

The aggregate is rebuildable from `metrics.jsonl`. Prompt targets and padding are excluded by the
explicit action mask. This report does not substitute teacher-forced loss for generated-answer
accuracy; run the linked evaluation profile against the saved adapter next.
"""
    write_text(run_dir / "report.md", markdown)
    escaped = html.escape(markdown)
    write_text(
        run_dir / "report.html",
        "<!doctype html><html><head><meta charset='utf-8'><title>NanoPT SFT Report</title>"
        "</head><body><pre>" + escaped + "</pre></body></html>\n",
    )


def execute_sft_run(
    result: ResolutionResult,
    *,
    tasks_path: Path,
    artifacts_root: Path,
    run_id: str | None,
    local_files_only: bool,
    device: str,
    resume_from: Path | None = None,
    train_limit: int | None = None,
) -> RunContext:
    """Load, adapt, train, validate, checkpoint, and report one SFT experiment."""

    experiment = result.config.experiment
    if not isinstance(experiment, SftExperiment):
        raise ValueError("SFT execution requires an sft experiment profile")
    if not experiment.data.completion_only:
        raise ValueError("NanoPT M4 implements completion-only SFT")
    if result.config.model.adapter is None:
        raise ValueError("model profile must define a LoRA adapter for SFT")

    all_tasks = read_arithmetic_tasks(tasks_path)
    dataset_manifest_path = tasks_path.with_name("dataset_manifest.json")
    dataset_manifest = read_split_manifest(dataset_manifest_path)
    validate_tasks_against_manifest(all_tasks, dataset_manifest)
    train_tasks, validation_tasks = _split_tasks(all_tasks, experiment)
    representative = train_limit is None
    if train_limit is not None:
        if train_limit <= 0:
            raise ValueError("train_limit must be positive")
        train_tasks = train_tasks[:train_limit]

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

        checkpoint_metadata = read_sft_checkpoint(resume_from) if resume_from else None
        if checkpoint_metadata is not None:
            if resume_from is None:  # Keeps the metadata/path relationship explicit to type tools.
                raise RuntimeError("checkpoint metadata requires a resume path")
            if checkpoint_metadata.adapter_name != experiment.adapter.name:
                raise ValueError("resume checkpoint adapter name differs from the SFT profile")
            policy = load_lora_adapter(
                loaded.model,
                resume_from / checkpoint_metadata.adapter_path,
                adapter_name=experiment.adapter.name,
                trainable=True,
            )
        else:
            policy = attach_lora_adapter(
                loaded.model,
                result.config.model.adapter,
                adapter_name=experiment.adapter.name,
            )
        selected_device = _select_device(device)
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
                "task_file_sha256": sha256_file(tasks_path),
                "split_manifest_sha256": sha256_file(dataset_manifest_path),
                "dataset": dataset_manifest.dataset_fingerprint,
            }
        )
        context.manifest["training"] = {
            "device": selected_device.type,
            "train_examples": len(train_tasks),
            "validation_examples": len(validation_tasks),
            "representative": representative,
            "resumed": resume_from is not None,
        }
        context.set_status("running")

        train_examples = render_sft_examples(train_tasks, renderer)
        validation_examples = render_sft_examples(validation_tasks, renderer)
        collator = CompletionOnlyCollator(
            pad_token_id=int(loaded.tokenizer.pad_token_id),
            max_sequence_length=experiment.data.max_sequence_length,
        )
        initial_nll, initial_accuracy, initial_tokens = evaluate_completion_nll(
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
                completion_nll=initial_nll,
                completion_token_accuracy=initial_accuracy,
                active_tokens=initial_tokens,
            ).model_dump(mode="json"),
        )

        optimizer = build_sft_optimizer(policy, experiment.training)
        starting_step = 0
        if checkpoint_metadata is not None and resume_from is not None:
            schedule = optimizer_groups(
                len(train_examples),
                micro_batch_size=experiment.training.micro_batch_size,
                gradient_accumulation_steps=experiment.training.gradient_accumulation_steps,
                epochs=experiment.training.epochs,
                seed=experiment.seed,
                max_steps=experiment.training.max_steps,
            )
            if checkpoint_metadata.total_optimizer_steps != len(schedule):
                raise ValueError("resume checkpoint was created for a different SFT schedule")
            restore_sft_optimizer(optimizer, resume_from, checkpoint_metadata)
            starting_step = checkpoint_metadata.optimizer_step

        latest_checkpoint: Path | None = None
        validation_values = (initial_nll, initial_accuracy, initial_tokens)

        def on_step(
            metric: SftStepMetrics,
            state: SftTrainingState,
            step_optimizer: torch.optim.Optimizer,
        ) -> None:
            nonlocal latest_checkpoint, validation_values
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
                validation_values = evaluate_completion_nll(
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
                        completion_nll=validation_values[0],
                        completion_token_accuracy=validation_values[1],
                        active_tokens=validation_values[2],
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
            starting_step=starting_step,
            optimizer=optimizer,
            on_step=on_step,
        )
        if training.state.optimizer_step == 0:
            raise ValueError("SFT run completed no optimizer steps")
        if latest_checkpoint is None:
            raise RuntimeError("SFT final optimizer boundary was not checkpointed")

        final_adapter_dir = save_lora_adapter(
            policy,
            context.run_dir / "adapter",
            adapter_name=experiment.adapter.name,
        )
        final_adapter_sha = sha256_directory(final_adapter_dir)
        context.manifest["model"]["adapter_sha256"] = final_adapter_sha
        context.manifest["checkpoint"] = {
            "path": final_adapter_dir.relative_to(context.run_dir).as_posix(),
            "sha256": final_adapter_sha,
            "parent_checkpoint_sha256": (
                checkpoint_metadata.adapter_sha256 if checkpoint_metadata else None
            ),
        }
        peak_allocated = max(
            (record.peak_allocated_bytes for record in training.metrics), default=0
        )
        peak_reserved = max((record.peak_reserved_bytes for record in training.metrics), default=0)
        summary = SftSummary(
            run_id=context.manifest["run_id"],
            optimizer_steps=training.state.optimizer_step,
            train_examples=len(train_examples),
            validation_examples=len(validation_examples),
            initial_validation_nll=initial_nll,
            final_validation_nll=validation_values[0],
            initial_validation_token_accuracy=initial_accuracy,
            final_validation_token_accuracy=validation_values[1],
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            representative=representative,
        )
        write_json(context.run_dir / "summary.json", summary.model_dump(mode="json"))
        _write_report(context.run_dir, summary)
        context.manifest["artifacts"] = [
            {
                "path": name,
                "kind": kind,
                "sha256": sha256_file(context.run_dir / name),
            }
            for name, kind in (
                ("metrics.jsonl", "sft_metrics"),
                ("summary.json", "sft_summary"),
                ("report.md", "markdown_report"),
                ("report.html", "html_report"),
            )
        ] + [
            {
                "path": final_adapter_dir.relative_to(context.run_dir).as_posix(),
                "kind": "lora_adapter",
                "sha256": final_adapter_sha,
            }
        ]
        context.set_status("completed")
        return context
    except Exception as exc:
        context.set_status(
            "failed",
            failure={"type": type(exc).__name__, "message": str(exc), "phase": "sft"},
        )
        raise
