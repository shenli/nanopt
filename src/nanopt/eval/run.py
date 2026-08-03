"""Public checkpoint-agnostic evaluation run orchestration used by CLI and pipelines."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import torch

from nanopt.config.loader import ConfigError
from nanopt.config.models import BaseEvalExperiment
from nanopt.config.resolver import ResolutionResult
from nanopt.eval.io import (
    read_arithmetic_tasks,
    read_split_manifest,
    validate_tasks_against_manifest,
)
from nanopt.eval.parser import answer_stop_token_ids
from nanopt.eval.runner import (
    EvaluationIdentity,
    EvaluationPlan,
    LocalModelBackend,
    evaluate_to_artifacts,
)
from nanopt.models.adapters import ParameterCounts, load_lora_adapter, parameter_counts
from nanopt.models.loading import LoadedModel, load_qwen3_base, qwen_chat_terminator_id
from nanopt.models.renderer import ChatRenderer
from nanopt.reporting.builder import build_evaluation_report
from nanopt.rollout.sampler import SamplingConfig
from nanopt.runtime.artifacts import sha256_file
from nanopt.runtime.run_context import RunContext, create_run_context
from nanopt.sft.checkpoint import sha256_directory


class EvaluationMode(StrEnum):
    deterministic = "deterministic"
    sampled = "sampled"


def evaluation_plan(
    experiment: BaseEvalExperiment,
    mode: EvaluationMode,
    *,
    eos_token_id: int,
    stop_token_sequence: tuple[int, ...],
) -> EvaluationPlan:
    """Translate a typed evaluation profile into the explicit sampler contract."""

    if mode is EvaluationMode.deterministic:
        deterministic = experiment.generation.deterministic
        sampling = SamplingConfig(
            max_new_tokens=deterministic.max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            stop_token_sequences=(stop_token_sequence,),
        )
        samples = 1
    else:
        sampled = experiment.generation.sampled
        sampling = SamplingConfig(
            max_new_tokens=sampled.max_new_tokens,
            do_sample=True,
            temperature=sampled.temperature,
            top_p=sampled.top_p,
            eos_token_id=eos_token_id,
            stop_token_sequences=(stop_token_sequence,),
        )
        samples = sampled.num_samples_per_prompt
    return EvaluationPlan(
        sampling=sampling,
        samples_per_task=samples,
        base_seed=experiment.seed,
        max_prompt_tokens=experiment.data.max_prompt_length,
    )


def move_model(loaded: LoadedModel, requested_device: str) -> str:
    """Move one loaded model to an explicitly validated CPU/CUDA destination."""

    if requested_device == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device in {"cpu", "cuda"}:
        selected = requested_device
    else:
        raise ValueError("device must be auto, cpu, or cuda")
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    loaded.model.to(selected)
    return selected


def _record_loaded_model(context: RunContext, loaded: LoadedModel, renderer: ChatRenderer) -> None:
    context.manifest["model"].update(
        {
            "resolved_revision": loaded.model_revision,
            "tokenizer_revision": loaded.tokenizer_revision,
            "chat_template_sha256": renderer.chat_template_sha256,
            "base_parameter_count": loaded.parameters.total,
            "trainable_parameter_count": loaded.parameters.trainable,
        }
    )
    context.set_status("running")


def _record_evaluation_artifacts(context: RunContext) -> None:
    context.manifest["artifacts"] = [
        {
            "path": name,
            "kind": kind,
            "sha256": sha256_file(context.run_dir / name),
        }
        for name, kind in (
            ("samples.jsonl", "evaluation_examples"),
            ("summary.json", "evaluation_summary"),
            ("report.md", "markdown_report"),
            ("report.html", "html_report"),
        )
    ]


def execute_evaluation_run(
    result: ResolutionResult,
    *,
    tasks_path: Path,
    mode: EvaluationMode,
    checkpoint_id: str,
    artifacts_root: Path,
    run_id: str | None,
    local_files_only: bool,
    device: str,
    limit: int | None,
    adapter_path: Path | None = None,
    adapter_name: str = "sft",
) -> RunContext:
    """Load an optional adapter, evaluate protected tasks, and persist inspectable artifacts."""

    experiment = result.config.experiment
    if not isinstance(experiment, BaseEvalExperiment):
        raise ConfigError("evaluation execution requires an evaluation experiment")
    all_tasks = read_arithmetic_tasks(tasks_path)
    dataset_manifest_path = tasks_path.with_name("dataset_manifest.json")
    dataset_manifest = read_split_manifest(dataset_manifest_path)
    validate_tasks_against_manifest(all_tasks, dataset_manifest)
    tasks = [task for task in all_tasks if task.split in experiment.data.splits]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("no task records match the evaluation profile splits")

    context = create_run_context(result, artifacts_root=artifacts_root, run_id=run_id)
    try:
        context.set_status("running")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        loaded = load_qwen3_base(result.config.model, local_files_only=local_files_only)
        if adapter_path is not None:
            base_parameter_count = loaded.parameters.total
            adapted = load_lora_adapter(
                loaded.model,
                adapter_path,
                adapter_name=adapter_name,
                trainable=False,
            )
            loaded = LoadedModel(
                model=adapted,
                tokenizer=loaded.tokenizer,
                model_revision=loaded.model_revision,
                tokenizer_revision=loaded.tokenizer_revision,
                parameters=ParameterCounts(
                    total=base_parameter_count,
                    trainable=parameter_counts(adapted).trainable,
                ),
            )
            context.manifest["model"]["adapter_name"] = adapter_name
            context.manifest["model"]["adapter_sha256"] = sha256_directory(adapter_path)
        selected_device = move_model(loaded, device)
        renderer = ChatRenderer(
            loaded.tokenizer,
            enable_thinking=result.config.model.renderer.enable_thinking,
            terminal_token_id=qwen_chat_terminator_id(loaded.tokenizer),
        )
        _record_loaded_model(context, loaded, renderer)
        terminal_token_id = qwen_chat_terminator_id(loaded.tokenizer)
        plan = evaluation_plan(
            experiment,
            mode,
            eos_token_id=terminal_token_id,
            stop_token_sequence=answer_stop_token_ids(loaded.tokenizer),
        )
        backend = LocalModelBackend(loaded.model, loaded.tokenizer, renderer)
        evaluate_to_artifacts(
            tasks,
            backend,
            EvaluationIdentity(context.manifest["run_id"], checkpoint_id),
            plan,
            samples_path=context.run_dir / "samples.jsonl",
            summary_path=context.run_dir / "summary.json",
        )
        build_evaluation_report(context.run_dir)
        context.manifest["data"]["fingerprints"]["task_file_sha256"] = sha256_file(tasks_path)
        context.manifest["data"]["fingerprints"]["split_manifest_sha256"] = sha256_file(
            dataset_manifest_path
        )
        context.manifest["data"]["fingerprints"]["dataset"] = dataset_manifest.dataset_fingerprint
        context.manifest["evaluation"] = {
            "mode": mode.value,
            "device": selected_device,
            "task_count": len(tasks),
            "representative": limit is None,
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if selected_device == "cuda" else 0
            ),
        }
        _record_evaluation_artifacts(context)
        context.set_status("completed")
        return context
    except Exception as exc:
        context.set_status(
            "failed",
            failure={"type": type(exc).__name__, "message": str(exc), "phase": "evaluation"},
        )
        raise
