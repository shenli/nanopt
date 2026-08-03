"""NanoPT command-line interface for inspectable configuration and evaluation."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import torch
import typer
import yaml
from rich.console import Console
from rich.table import Table

from nanopt.config.loader import ConfigError, ConfigRepository
from nanopt.config.models import (
    BaseEvalExperiment,
    DpoExperiment,
    GrpoExperiment,
    SftExperiment,
)
from nanopt.config.provenance import serialize_provenance
from nanopt.config.resolver import ResolutionResult, resolve_config
from nanopt.data.arithmetic import ArithmeticGeneratorConfig, generate_tasks
from nanopt.data.preferences import generate_preference_pairs
from nanopt.data.schemas import ArithmeticSplitConfig
from nanopt.data.splits import SPLIT_ORDER, build_splits
from nanopt.dpo.run import execute_dpo_run
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
from nanopt.grpo.run import execute_grpo_run
from nanopt.models.adapters import ParameterCounts, load_lora_adapter, parameter_counts
from nanopt.models.loading import LoadedModel, load_qwen3_base, qwen_chat_terminator_id
from nanopt.models.renderer import ChatRenderer
from nanopt.reporting.builder import build_evaluation_report
from nanopt.rollout.sampler import SamplingConfig
from nanopt.runtime.artifacts import append_jsonl, sha256_file, write_json, write_yaml
from nanopt.runtime.doctor import DoctorReport, collect_doctor_report
from nanopt.runtime.run_context import RunContext, create_run_context
from nanopt.sft.checkpoint import sha256_directory
from nanopt.sft.run import execute_sft_run
from nanopt.version import __version__

app = typer.Typer(
    name="nanopt",
    help="NanoPT: a white-box course and reference implementation for post-training.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(help="Validate and deterministically resolve configuration profiles.")
artifacts_app = typer.Typer(help="Inspect local NanoPT run artifacts.")
data_app = typer.Typer(help="Generate and validate deterministic local task artifacts.")
eval_app = typer.Typer(help="Run checkpoint-agnostic generation evaluation.")
report_app = typer.Typer(help="Build local reports from inspectable run artifacts.")
train_app = typer.Typer(help="Run readable white-box training stages.")
app.add_typer(config_app, name="config")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(report_app, name="report")
app.add_typer(train_app, name="train")
console = Console()


class EvaluationMode(StrEnum):
    deterministic = "deterministic"
    sampled = "sampled"


class CalibrationMode(StrEnum):
    load = "load"
    eval = "eval"
    sft = "sft"
    dpo = "dpo"
    grpo = "grpo"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"nanopt {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """Expose implemented, inspectable NanoPT commands."""


@app.command("version")
def version_command() -> None:
    """Print the installed NanoPT version."""

    console.print(__version__)


def _doctor_table(report: DoctorReport) -> Table:
    table = Table(title="NanoPT environment diagnosis", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Status", f"{report.status} (exit {report.exit_code})")
    table.add_row("Platform", f"{report.os}/{report.architecture}")
    table.add_row("Python", report.python_version)
    table.add_row("PyTorch", report.pytorch_version or "missing")
    table.add_row("CUDA", report.cuda.runtime_version or "unavailable")
    table.add_row("CUDA devices", str(report.cuda.device_count))
    for gpu in report.cuda.gpus:
        gib = gpu.total_vram_bytes / 1024**3
        free_gib = gpu.free_vram_bytes / 1024**3
        table.add_row(
            f"GPU {gpu.index}",
            f"{gpu.name}; {free_gib:.2f}/{gib:.2f} GiB free; CC {gpu.compute_capability}",
        )
    table.add_row(
        "Docker",
        report.docker.version
        if report.docker.daemon_reachable
        else ("daemon unavailable" if report.docker.executable_found else "not installed"),
    )
    if report.profile.requested_id:
        label = "match" if report.profile.matched else "mismatch"
        table.add_row("Hardware profile", f"{report.profile.requested_id}: {label}")
    for message in report.messages:
        table.add_row("Notice", message)
    return table


@app.command()
def doctor(
    json_path: Annotated[
        Path | None,
        typer.Option("--json", help="Write the complete machine-readable report to this path."),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option(help="Known hardware profile to compare against."),
    ] = "rtx_4070_ti_super_16gb",
    strict_profile: Annotated[
        bool,
        typer.Option("--strict-profile", help="Treat a requested profile mismatch as exit code 4."),
    ] = False,
    config_dir: Annotated[
        Path | None,
        typer.Option(help="Override the canonical profile directory."),
    ] = None,
) -> None:
    """Inspect dependencies, CUDA, GPU memory, BF16, Docker, and profile compatibility."""

    repository = ConfigRepository(config_dir) if config_dir else ConfigRepository()
    try:
        hardware = repository.hardware(profile) if profile else None
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    report = collect_doctor_report(hardware, strict_profile=strict_profile)
    if json_path is not None:
        write_json(json_path, report.model_dump(mode="json"))
    console.print(_doctor_table(report))
    if report.exit_code:
        raise typer.Exit(code=report.exit_code)


@config_app.command("resolve")
def config_resolve(
    hardware: Annotated[str, typer.Option(help="Hardware profile ID.")] = (
        "rtx_4070_ti_super_16gb"
    ),
    model: Annotated[str, typer.Option(help="Model profile ID.")] = "qwen3_0_6b_base",
    experiment: Annotated[str | None, typer.Option(help="Experiment profile ID.")] = "base_eval",
    recipe: Annotated[str | None, typer.Option(help="Optional recipe profile ID.")] = None,
    stage: Annotated[str | None, typer.Option(help="Optional stage within --recipe.")] = None,
    set_values: Annotated[
        list[str] | None,
        typer.Option(
            "--set", help="Dotted scalar override (repeatable), for example rollout.group_size=2."
        ),
    ] = None,
    output: Annotated[Path, typer.Option(help="Stable resolved YAML output path.")] = Path(
        "resolved_config.yaml"
    ),
    config_dir: Annotated[
        Path | None,
        typer.Option(help="Override the canonical profile directory."),
    ] = None,
) -> None:
    """Resolve strict profiles and show the source of every resulting value."""

    repository = ConfigRepository(config_dir) if config_dir else ConfigRepository()
    try:
        result = resolve_config(
            repository=repository,
            hardware_id=hardware,
            model_id=model,
            experiment_id=experiment,
            recipe_id=recipe,
            recipe_stage_id=stage,
            overrides=tuple(set_values or ()),
        )
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc

    value = result.config.model_dump(mode="json", exclude_none=False)
    write_yaml(output, value)
    provenance_path = output.with_name(f"{output.stem}.provenance.yaml")
    write_yaml(provenance_path, serialize_provenance(result.provenance))

    table = Table(title="Resolved NanoPT configuration")
    table.add_column("Value")
    table.add_column("Resolved ID")
    table.add_column("Source")
    table.add_row("hardware", result.config.hardware.id, f"hardware:{result.config.hardware.id}")
    table.add_row("model", result.config.model.id, f"model:{result.config.model.id}")
    table.add_row(
        "experiment", result.config.experiment.id, f"experiment:{result.config.experiment.id}"
    )
    if result.config.recipe:
        table.add_row("recipe", result.config.recipe.id, f"recipe:{result.config.recipe.id}")
    for expression in result.cli_overrides:
        table.add_row("override", expression, "cli_override")
    console.print(table)
    console.print(f"Wrote [bold]{output}[/bold] and [bold]{provenance_path}[/bold]")


@artifacts_app.command("inspect")
def artifacts_inspect(
    run_dir: Annotated[Path, typer.Argument(help="Run directory containing run_manifest.json.")],
) -> None:
    """Inspect a run manifest without loading a model."""

    manifest_path = run_dir / "run_manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        console.print(f"[red]Missing manifest:[/red] {manifest_path}")
        raise typer.Exit(code=1) from exc
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid manifest JSON:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(yaml.safe_dump(value, allow_unicode=True, sort_keys=True), highlight=False)


@data_app.command("generate")
def data_generate(
    generator_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Arithmetic generator YAML."),
    ] = Path("tasks/arithmetic/generator_config.yaml"),
    split_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Leakage-safe split YAML."),
    ] = Path("tasks/arithmetic/split_config.yaml"),
    output: Annotated[Path, typer.Option(help="New or empty task JSONL output.")] = Path(
        "artifacts/data/arithmetic_v1/tasks.jsonl"
    ),
    manifest: Annotated[Path | None, typer.Option(help="Optional split manifest path.")] = None,
) -> None:
    """Generate fingerprinted arithmetic tasks and assign every task to one split."""

    try:
        generator_value = yaml.safe_load(generator_config.read_text(encoding="utf-8"))
        split_value = yaml.safe_load(split_config.read_text(encoding="utf-8"))
        generator = ArithmeticGeneratorConfig.model_validate(generator_value, strict=True)
        split = ArithmeticSplitConfig.model_validate(split_value, strict=True)
        if output.exists() and output.stat().st_size:
            raise ValueError("output must be new or empty to prevent mixed dataset versions")
        manifest_path = manifest or output.with_name("dataset_manifest.json")
        if manifest_path.resolve() == output.resolve():
            raise ValueError("manifest and task output paths must be different")
        generated = generate_tasks(generator)
        splits, split_manifest = build_splits(
            generated,
            counts=split.counts,
            seed=split.seed,
            generator_config=generator,
        )
        for name in SPLIT_ORDER:
            for task in splits[name]:
                append_jsonl(output, task.model_dump(mode="json", exclude_none=True))
        write_json(manifest_path, split_manifest.model_dump(mode="json"))
    except (OSError, TypeError, ValueError) as exc:
        console.print(f"[red]Data generation failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(
        f"Wrote {generator.count} tasks to [bold]{output}[/bold]; "
        f"fingerprint {split_manifest.dataset_fingerprint}"
    )


@data_app.command("preferences")
def data_preferences(
    tasks: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Source arithmetic task JSONL.")
    ] = Path("artifacts/data/arithmetic_v1/tasks.jsonl"),
    output: Annotated[Path, typer.Option(help="New or empty preference JSONL output.")] = Path(
        "artifacts/data/arithmetic_preferences_v1/preferences.jsonl"
    ),
    audit: Annotated[Path | None, typer.Option(help="Optional preference audit path.")] = None,
    seed: Annotated[int, typer.Option(help="Controlled rejection assignment seed.")] = 42,
) -> None:
    """Construct verifier-audited chosen/rejected pairs from non-protected task splits."""

    try:
        if output.exists() and output.stat().st_size:
            raise ValueError("preference output must be new or empty")
        task_records = read_arithmetic_tasks(tasks)
        source_manifest_path = tasks.with_name("dataset_manifest.json")
        source_manifest = read_split_manifest(source_manifest_path)
        validate_tasks_against_manifest(task_records, source_manifest)
        pairs, preference_audit = generate_preference_pairs(
            task_records,
            source_dataset_fingerprint=source_manifest.dataset_fingerprint,
            seed=seed,
        )
        for pair in pairs:
            append_jsonl(output, pair.model_dump(mode="json"))
        audit_path = audit or output.with_name("preference_audit.json")
        write_json(audit_path, preference_audit.model_dump(mode="json"))
    except (OSError, TypeError, ValueError) as exc:
        console.print(f"[red]Preference construction failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(
        f"Wrote {len(pairs)} controlled pairs to [bold]{output}[/bold]; "
        f"fingerprint {preference_audit.dataset_fingerprint}"
    )


def _resolve_evaluation(
    *,
    hardware: str,
    model: str,
    experiment: str,
    config_dir: Path | None,
) -> ResolutionResult:
    repository = ConfigRepository(config_dir) if config_dir else ConfigRepository()
    result = resolve_config(
        repository=repository,
        hardware_id=hardware,
        model_id=model,
        experiment_id=experiment,
    )
    if not isinstance(result.config.experiment, BaseEvalExperiment):
        raise ConfigError(f"experiment {experiment!r} is not an evaluation profile")
    return result


def _resolve_sft(
    *,
    hardware: str,
    model: str,
    experiment: str,
    config_dir: Path | None,
    overrides: tuple[str, ...] = (),
) -> ResolutionResult:
    repository = ConfigRepository(config_dir) if config_dir else ConfigRepository()
    result = resolve_config(
        repository=repository,
        hardware_id=hardware,
        model_id=model,
        experiment_id=experiment,
        overrides=overrides,
    )
    if not isinstance(result.config.experiment, SftExperiment):
        raise ConfigError(f"experiment {experiment!r} is not an SFT profile")
    return result


def _resolve_dpo(
    *,
    hardware: str,
    model: str,
    experiment: str,
    config_dir: Path | None,
    overrides: tuple[str, ...] = (),
) -> ResolutionResult:
    repository = ConfigRepository(config_dir) if config_dir else ConfigRepository()
    result = resolve_config(
        repository=repository,
        hardware_id=hardware,
        model_id=model,
        experiment_id=experiment,
        overrides=overrides,
    )
    if not isinstance(result.config.experiment, DpoExperiment):
        raise ConfigError(f"experiment {experiment!r} is not a DPO profile")
    return result


def _resolve_grpo(
    *,
    hardware: str,
    model: str,
    experiment: str,
    config_dir: Path | None,
    overrides: tuple[str, ...] = (),
) -> ResolutionResult:
    repository = ConfigRepository(config_dir) if config_dir else ConfigRepository()
    result = resolve_config(
        repository=repository,
        hardware_id=hardware,
        model_id=model,
        experiment_id=experiment,
        overrides=overrides,
    )
    if not isinstance(result.config.experiment, GrpoExperiment):
        raise ConfigError(f"experiment {experiment!r} is not a GRPO profile")
    return result


def _evaluation_plan(
    experiment: BaseEvalExperiment,
    mode: EvaluationMode,
    *,
    eos_token_id: int,
    stop_token_sequence: tuple[int, ...],
) -> EvaluationPlan:
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


def _move_model(loaded: LoadedModel, requested_device: str) -> str:
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


def _execute_evaluation(
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
        selected_device = _move_model(loaded, device)
        renderer = ChatRenderer(
            loaded.tokenizer,
            enable_thinking=result.config.model.renderer.enable_thinking,
            terminal_token_id=qwen_chat_terminator_id(loaded.tokenizer),
        )
        _record_loaded_model(context, loaded, renderer)
        terminal_token_id = qwen_chat_terminator_id(loaded.tokenizer)
        plan = _evaluation_plan(
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


@eval_app.command("run")
def eval_run(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Task JSONL file.")],
    mode: Annotated[EvaluationMode, typer.Option(help="Deterministic or sampled evaluation.")] = (
        EvaluationMode.deterministic
    ),
    checkpoint_id: Annotated[str, typer.Option(help="Checkpoint identity recorded in results.")] = (
        "base"
    ),
    hardware: Annotated[str, typer.Option(help="Hardware profile ID.")] = (
        "rtx_4070_ti_super_16gb"
    ),
    model: Annotated[str, typer.Option(help="Model profile ID.")] = "qwen3_0_6b_base",
    experiment: Annotated[str, typer.Option(help="Evaluation experiment profile ID.")] = (
        "base_eval"
    ),
    artifacts_root: Annotated[Path, typer.Option(help="Parent directory for the new run.")] = Path(
        "artifacts/runs"
    ),
    run_id: Annotated[str | None, typer.Option(help="Optional path-safe run ID.")] = None,
    local_files_only: Annotated[
        bool, typer.Option(help="Forbid Hugging Face network access and use cached files only.")
    ] = False,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda.")] = "auto",
    adapter: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Optional local LoRA adapter directory."),
    ] = None,
    adapter_name: Annotated[str, typer.Option(help="Name assigned to --adapter.")] = "sft",
    config_dir: Annotated[Path | None, typer.Option(help="Override profile directory.")] = None,
) -> None:
    """Evaluate one checkpoint and write examples before aggregates and reports."""

    try:
        resolved = _resolve_evaluation(
            hardware=hardware,
            model=model,
            experiment=experiment,
            config_dir=config_dir,
        )
        context = _execute_evaluation(
            resolved,
            tasks_path=tasks,
            mode=mode,
            checkpoint_id=checkpoint_id,
            artifacts_root=artifacts_root,
            run_id=run_id,
            local_files_only=local_files_only,
            device=device,
            limit=None,
            adapter_path=adapter,
            adapter_name=adapter_name,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]Evaluation failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Evaluation completed: [bold]{context.run_dir}[/bold]")


@train_app.command("sft")
def train_sft_command(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Task JSONL file.")],
    hardware: Annotated[str, typer.Option(help="Hardware profile ID.")] = (
        "rtx_4070_ti_super_16gb"
    ),
    model: Annotated[str, typer.Option(help="Model profile ID.")] = "qwen3_0_6b_base",
    experiment: Annotated[str, typer.Option(help="SFT experiment profile ID.")] = "math_sft",
    artifacts_root: Annotated[Path, typer.Option(help="Parent directory for the new run.")] = Path(
        "artifacts/runs"
    ),
    run_id: Annotated[str | None, typer.Option(help="Optional path-safe SFT run ID.")] = None,
    resume_from: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Clean-boundary SFT checkpoint directory."),
    ] = None,
    local_files_only: Annotated[bool, typer.Option()] = False,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda.")] = "auto",
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Repeatable scalar override within the SFT profile."),
    ] = None,
    config_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train one completion-only LoRA adapter with explicit optimizer semantics."""

    try:
        resolved = _resolve_sft(
            hardware=hardware,
            model=model,
            experiment=experiment,
            config_dir=config_dir,
            overrides=tuple(set_values or ()),
        )
        context = execute_sft_run(
            resolved,
            tasks_path=tasks,
            artifacts_root=artifacts_root,
            run_id=run_id,
            local_files_only=local_files_only,
            device=device,
            resume_from=resume_from,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]SFT failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"SFT completed: [bold]{context.run_dir}[/bold]")


@train_app.command("dpo")
def train_dpo_command(
    preferences: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Preference JSONL file.")
    ],
    sft_adapter: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen parent SFT adapter directory."),
    ],
    hardware: Annotated[str, typer.Option()] = "rtx_4070_ti_super_16gb",
    model: Annotated[str, typer.Option()] = "qwen3_0_6b_base",
    experiment: Annotated[str, typer.Option()] = "math_dpo",
    artifacts_root: Annotated[Path, typer.Option()] = Path("artifacts/runs"),
    run_id: Annotated[str | None, typer.Option(help="Optional path-safe DPO run ID.")] = None,
    local_files_only: Annotated[bool, typer.Option()] = False,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda.")] = "auto",
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Repeatable scalar override within the DPO profile."),
    ] = None,
    config_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train a DPO adapter from an exact SFT copy and a fingerprinted reference cache."""

    try:
        resolved = _resolve_dpo(
            hardware=hardware,
            model=model,
            experiment=experiment,
            config_dir=config_dir,
            overrides=tuple(set_values or ()),
        )
        context = execute_dpo_run(
            resolved,
            preferences_path=preferences,
            sft_adapter_path=sft_adapter,
            artifacts_root=artifacts_root,
            run_id=run_id,
            local_files_only=local_files_only,
            device=device,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]DPO failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"DPO completed: [bold]{context.run_dir}[/bold]")


@train_app.command("grpo")
def train_grpo_command(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Task JSONL file.")],
    dpo_adapter: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen parent DPO adapter directory."),
    ],
    hardware: Annotated[str, typer.Option()] = "rtx_4070_ti_super_16gb",
    model: Annotated[str, typer.Option()] = "qwen3_0_6b_base",
    experiment: Annotated[str, typer.Option()] = "math_grpo",
    artifacts_root: Annotated[Path, typer.Option()] = Path("artifacts/runs"),
    run_id: Annotated[str | None, typer.Option(help="Optional path-safe GRPO run ID.")] = None,
    iteration_limit: Annotated[
        int | None,
        typer.Option(help="Explicit non-representative iteration cap for recipe pilots."),
    ] = None,
    local_files_only: Annotated[bool, typer.Option()] = False,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda.")] = "auto",
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Repeatable scalar override within the GRPO profile."),
    ] = None,
    config_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Run fresh grouped RLVR rollouts and exact-token synchronous GRPO updates."""

    try:
        resolved = _resolve_grpo(
            hardware=hardware,
            model=model,
            experiment=experiment,
            config_dir=config_dir,
            overrides=tuple(set_values or ()),
        )
        context = execute_grpo_run(
            resolved,
            tasks_path=tasks,
            dpo_adapter_path=dpo_adapter,
            artifacts_root=artifacts_root,
            run_id=run_id,
            local_files_only=local_files_only,
            device=device,
            iteration_limit=iteration_limit,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]GRPO failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"GRPO completed: [bold]{context.run_dir}[/bold]")


@report_app.command("build")
def report_build(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Rebuild Markdown, HTML, and summary artifacts without loading a model."""

    try:
        artifacts = build_evaluation_report(run_dir)
    except (OSError, TypeError, ValueError) as exc:
        console.print(f"[red]Report build failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(
        f"Wrote [bold]{artifacts.markdown}[/bold], [bold]{artifacts.html}[/bold], "
        f"and [bold]{artifacts.summary}[/bold]"
    )


@app.command()
def calibrate(
    mode: Annotated[CalibrationMode, typer.Option(help="Load-only or short evaluation path.")],
    tasks: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Required task JSONL for --mode eval."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(help="Explicit non-representative example limit for eval calibration."),
    ] = 2,
    hardware: Annotated[str, typer.Option()] = "rtx_4070_ti_super_16gb",
    model: Annotated[str, typer.Option()] = "qwen3_0_6b_base",
    experiment: Annotated[str, typer.Option()] = "base_eval",
    sft_experiment: Annotated[str, typer.Option(help="SFT profile for --mode sft.")] = "math_sft",
    dpo_experiment: Annotated[str, typer.Option(help="DPO profile for --mode dpo.")] = "math_dpo",
    grpo_experiment: Annotated[str, typer.Option(help="GRPO profile for --mode grpo.")] = (
        "math_grpo"
    ),
    preferences: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Required preference JSONL for DPO."),
    ] = None,
    sft_adapter: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Required SFT adapter for DPO."),
    ] = None,
    dpo_adapter: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Required DPO adapter for GRPO."),
    ] = None,
    artifacts_root: Annotated[Path, typer.Option()] = Path("artifacts/runs"),
    run_id: Annotated[str | None, typer.Option(help="Optional path-safe eval calibration ID.")] = (
        None
    ),
    local_files_only: Annotated[bool, typer.Option()] = False,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda.")] = "auto",
    config_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Exercise the real model load or evaluation path before a full reference run."""

    try:
        if mode is CalibrationMode.grpo:
            if tasks is None or dpo_adapter is None:
                raise ValueError("--tasks and --dpo-adapter are required for --mode grpo")
            grpo_resolved = _resolve_grpo(
                hardware=hardware,
                model=model,
                experiment=grpo_experiment,
                config_dir=config_dir,
                overrides=(
                    "rollout.group_size=2",
                    "optimization.iterations=1",
                    "optimization.minibatch_completions=2",
                    "optimization.gradient_accumulation_steps=1",
                ),
            )
            context = execute_grpo_run(
                grpo_resolved,
                tasks_path=tasks,
                dpo_adapter_path=dpo_adapter,
                artifacts_root=artifacts_root,
                run_id=run_id,
                local_files_only=local_files_only,
                device=device,
                iteration_limit=1,
            )
            console.print(
                f"Non-representative GRPO calibration completed: [bold]{context.run_dir}[/bold]"
            )
            return
        if mode is CalibrationMode.dpo:
            if preferences is None or sft_adapter is None:
                raise ValueError("--preferences and --sft-adapter are required for --mode dpo")
            dpo_resolved = _resolve_dpo(
                hardware=hardware,
                model=model,
                experiment=dpo_experiment,
                config_dir=config_dir,
                overrides=(
                    "training.pair_micro_batch_size=1",
                    "training.gradient_accumulation_steps=1",
                    "reference.cache_validation_sample_size=1",
                ),
            )
            context = execute_dpo_run(
                dpo_resolved,
                preferences_path=preferences,
                sft_adapter_path=sft_adapter,
                artifacts_root=artifacts_root,
                run_id=run_id,
                local_files_only=local_files_only,
                device=device,
                pair_limit=limit,
            )
            console.print(
                f"Non-representative DPO calibration completed: [bold]{context.run_dir}[/bold]"
            )
            return
        if mode is CalibrationMode.sft:
            if tasks is None:
                raise ValueError("--tasks is required for --mode sft")
            sft_resolved = _resolve_sft(
                hardware=hardware,
                model=model,
                experiment=sft_experiment,
                config_dir=config_dir,
                overrides=(
                    "training.micro_batch_size=1",
                    "training.gradient_accumulation_steps=1",
                    "training.max_steps=1",
                ),
            )
            context = execute_sft_run(
                sft_resolved,
                tasks_path=tasks,
                artifacts_root=artifacts_root,
                run_id=run_id,
                local_files_only=local_files_only,
                device=device,
                train_limit=limit,
            )
            console.print(
                f"Non-representative SFT calibration completed: [bold]{context.run_dir}[/bold]"
            )
            return
        resolved = _resolve_evaluation(
            hardware=hardware,
            model=model,
            experiment=experiment,
            config_dir=config_dir,
        )
        if mode is CalibrationMode.load:
            loaded = load_qwen3_base(resolved.config.model, local_files_only=local_files_only)
            selected = _move_model(loaded, device)
            console.print(
                f"Loaded {resolved.config.model.source.model_id} on {selected}; "
                f"{loaded.parameters.total:,} parameters; revision {loaded.model_revision}"
            )
            return
        if tasks is None:
            raise ValueError("--tasks is required for --mode eval")
        context = _execute_evaluation(
            resolved,
            tasks_path=tasks,
            mode=EvaluationMode.deterministic,
            checkpoint_id="base-calibration",
            artifacts_root=artifacts_root,
            run_id=run_id,
            local_files_only=local_files_only,
            device=device,
            limit=limit,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]Calibration failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Non-representative calibration completed: [bold]{context.run_dir}[/bold]")
