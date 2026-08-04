"""NanoPT command-line interface for inspectable configuration and evaluation."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from nanopt.agent.rl_run import execute_agent_rl_run
from nanopt.agent.run import execute_agent_run
from nanopt.agent.sft_data import build_agent_sft_dataset
from nanopt.agent.sft_run import execute_agent_sft_run
from nanopt.config.loader import ConfigError, ConfigRepository
from nanopt.config.models import (
    AgentEvaluationExperiment,
    AgentRlExperiment,
    AgentSftExperiment,
    BaseEvalExperiment,
    DpoExperiment,
    GrpoExperiment,
    SftExperiment,
    SystemsLabExperiment,
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
from nanopt.eval.run import EvaluationMode, execute_evaluation_run, move_model
from nanopt.grpo.run import execute_grpo_run
from nanopt.models.loading import load_qwen3_base
from nanopt.pipeline.run import execute_pipeline
from nanopt.reporting.builder import build_evaluation_report
from nanopt.runtime.artifacts import append_jsonl, write_json, write_yaml
from nanopt.runtime.doctor import DoctorReport, collect_doctor_report
from nanopt.sft.run import execute_sft_run
from nanopt.systems.run import execute_systems_lab_run
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
pipeline_app = typer.Typer(help="Run or resume the explicit Base-to-GRPO recipe.")
agent_app = typer.Typer(help="Evaluate structured policies in the resettable MiniSWE environment.")
systems_app = typer.Typer(help="Run deterministic rollout-infrastructure teaching simulations.")
app.add_typer(config_app, name="config")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(report_app, name="report")
app.add_typer(train_app, name="train")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(agent_app, name="agent")
app.add_typer(systems_app, name="systems")
console = Console()


class CalibrationMode(StrEnum):
    load = "load"
    eval = "eval"
    sft = "sft"
    dpo = "dpo"
    grpo = "grpo"


class AgentPolicyMode(StrEnum):
    oracle = "oracle"
    model = "model"


class AgentContextMode(StrEnum):
    observation_snapshot = "observation_snapshot"
    full_transcript = "full_transcript"


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


def _resolve_agent(
    *,
    hardware: str,
    model: str,
    experiment: str,
    backend: str,
    task_split: str,
    context_policy: str,
    config_dir: Path | None,
) -> ResolutionResult:
    repository = ConfigRepository(config_dir) if config_dir else ConfigRepository()
    result = resolve_config(
        repository=repository,
        hardware_id=hardware,
        model_id=model,
        experiment_id=experiment,
        overrides=(
            f"environment.backend={backend}",
            f"tasks.split={task_split}",
            f"policy.context_policy={context_policy}",
        ),
    )
    if not isinstance(result.config.experiment, AgentEvaluationExperiment):
        raise ConfigError(f"experiment {experiment!r} is not an agent-evaluation profile")
    return result


def _resolve_agent_sft(
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
    if not isinstance(result.config.experiment, AgentSftExperiment):
        raise ConfigError(f"experiment {experiment!r} is not an Agent SFT profile")
    return result


def _resolve_agent_rl(
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
    if not isinstance(result.config.experiment, AgentRlExperiment):
        raise ConfigError(f"experiment {experiment!r} is not an Agent RL profile")
    return result


def _resolve_systems_lab(
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
    if not isinstance(result.config.experiment, SystemsLabExperiment):
        raise ConfigError(f"experiment {experiment!r} is not a systems-lab profile")
    return result


@systems_app.command("simulate")
def systems_simulate_command(
    hardware: Annotated[str, typer.Option(help="Metadata hardware profile ID.")] = (
        "rtx_4070_ti_super_16gb"
    ),
    model: Annotated[str, typer.Option(help="Metadata model profile ID.")] = "qwen3_0_6b_base",
    experiment: Annotated[
        str, typer.Option(help="Systems-lab experiment profile ID.")
    ] = "resumable_rollouts",
    artifacts_root: Annotated[Path, typer.Option(help="Parent directory for the run.")] = Path(
        "artifacts/runs"
    ),
    run_id: Annotated[str | None, typer.Option(help="Optional path-safe run ID.")] = None,
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Repeatable scalar override within the systems profile."),
    ] = None,
    config_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Compare resumable-rollout weight, cache, and freshness policies on CPU."""

    try:
        resolved = _resolve_systems_lab(
            hardware=hardware,
            model=model,
            experiment=experiment,
            config_dir=config_dir,
            overrides=tuple(set_values or ()),
        )
        context = execute_systems_lab_run(
            resolved,
            artifacts_root=artifacts_root,
            run_id=run_id,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]Systems simulation failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Systems simulation completed: [bold]{context.run_dir}[/bold]")


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
        context = execute_evaluation_run(
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


@pipeline_app.command("run")
def pipeline_run_command(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Task JSONL file.")],
    recipe: Annotated[str, typer.Option(help="Recipe profile ID.")] = "math_pipeline",
    artifacts_root: Annotated[
        Path, typer.Option(help="Parent directory for the pipeline run.")
    ] = Path("artifacts/pipelines"),
    run_id: Annotated[str | None, typer.Option(help="Stable path-safe pipeline run ID.")] = None,
    resume: Annotated[
        bool, typer.Option(help="Resume and hash-check an existing --run-id.")
    ] = False,
    local_files_only: Annotated[
        bool, typer.Option(help="Forbid model downloads and use the local cache only.")
    ] = False,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda.")] = "auto",
    config_dir: Annotated[Path | None, typer.Option(help="Override profile directory.")] = None,
) -> None:
    """Run calibrations and Base -> SFT -> DPO -> GRPO as visible child runs."""

    if resume and run_id is None:
        console.print("[red]Pipeline failed:[/red] --resume requires --run-id")
        raise typer.Exit(code=1)
    try:
        pipeline_dir = execute_pipeline(
            tasks_path=tasks,
            artifacts_root=artifacts_root,
            recipe_id=recipe,
            config_dir=config_dir,
            run_id=run_id,
            resume=resume,
            local_files_only=local_files_only,
            device=device,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]Pipeline failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Pipeline completed: [bold]{pipeline_dir}[/bold]")


@agent_app.command("run")
def agent_run_command(
    tasks_root: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="MiniSWE suite root containing suite.yaml."
        ),
    ] = Path("tasks/mini_swe_v1"),
    policy: Annotated[AgentPolicyMode, typer.Option(help="Scripted oracle or Qwen policy.")] = (
        AgentPolicyMode.oracle
    ),
    backend: Annotated[
        str, typer.Option(help="docker (secure reference) or fake (trusted tests).")
    ] = "docker",
    task_split: Annotated[str, typer.Option(help="smoke, reference, or all.")] = "smoke",
    context_policy: Annotated[
        AgentContextMode,
        typer.Option(help="Snapshot history or alternating full-transcript messages."),
    ] = AgentContextMode.observation_snapshot,
    task_id: Annotated[
        list[str] | None,
        typer.Option("--task-id", help="Run only this task ID; repeat to select several."),
    ] = None,
    hardware: Annotated[str, typer.Option(help="Hardware profile ID.")] = (
        "rtx_4070_ti_super_16gb"
    ),
    model: Annotated[str, typer.Option(help="Model profile ID.")] = "qwen3_0_6b_base",
    experiment: Annotated[str, typer.Option(help="Agent experiment profile ID.")] = (
        "mini_swe_rollout"
    ),
    artifacts_root: Annotated[Path, typer.Option(help="Parent directory for the run.")] = Path(
        "artifacts/runs"
    ),
    run_id: Annotated[str | None, typer.Option(help="Optional path-safe run ID.")] = None,
    adapter: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Optional policy LoRA adapter."),
    ] = None,
    adapter_name: Annotated[str, typer.Option(help="Name assigned to --adapter.")] = "grpo",
    max_tasks: Annotated[
        int | None, typer.Option(help="Non-representative prefix limit for smoke/model runs.")
    ] = None,
    turn_limit: Annotated[
        int | None, typer.Option(help="Non-representative per-task turn limit.")
    ] = None,
    local_files_only: Annotated[
        bool, typer.Option(help="Forbid model downloads and use the local cache only.")
    ] = False,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda for model policy only.")] = "auto",
    config_dir: Annotated[Path | None, typer.Option(help="Override profile directory.")] = None,
) -> None:
    """Evaluate a structured policy in resettable MiniSWE workspaces."""

    try:
        if backend not in {"docker", "fake"}:
            raise ValueError("backend must be docker or fake")
        resolved = _resolve_agent(
            hardware=hardware,
            model=model,
            experiment=experiment,
            backend=backend,
            task_split=task_split,
            context_policy=context_policy.value,
            config_dir=config_dir,
        )
        context = execute_agent_run(
            resolved,
            tasks_root=tasks_root,
            policy_kind=policy.value,
            artifacts_root=artifacts_root,
            run_id=run_id,
            adapter_path=adapter,
            adapter_name=adapter_name,
            local_files_only=local_files_only,
            device=device,
            max_tasks=max_tasks,
            turn_limit=turn_limit,
            task_ids=tuple(task_id or ()),
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]Agent evaluation failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Agent evaluation completed: [bold]{context.run_dir}[/bold]")


@agent_app.command("build-sft-data")
def agent_build_sft_data_command(
    tasks_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="MiniSWE suite root."),
    ] = Path("tasks/mini_swe_v1"),
    output: Annotated[
        Path,
        typer.Option(help="New directory for examples, manifest, and source trajectories."),
    ] = Path("artifacts/data/mini_swe_agent_sft_v1"),
    hardware: Annotated[str, typer.Option(help="Hardware profile ID.")] = (
        "rtx_4070_ti_super_16gb"
    ),
    model: Annotated[str, typer.Option(help="Model/tokenizer profile ID.")] = "qwen3_0_6b_base",
    experiment: Annotated[str, typer.Option(help="Agent SFT experiment profile ID.")] = "agent_sft",
    local_files_only: Annotated[
        bool, typer.Option(help="Use only the locally cached pinned tokenizer.")
    ] = False,
    config_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Collect replay-checked demonstrations and freeze exact token/action masks."""

    try:
        resolved = _resolve_agent_sft(
            hardware=hardware,
            model=model,
            experiment=experiment,
            config_dir=config_dir,
        )
        profile = resolved.config.experiment
        if not isinstance(profile, AgentSftExperiment):
            raise AssertionError("Agent SFT profile discriminator mismatch")
        manifest = build_agent_sft_dataset(
            profile,
            resolved.config.model,
            tasks_root=tasks_root,
            output_dir=output,
            local_files_only=local_files_only,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]Agent SFT data build failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(
        f"Agent SFT dataset completed: [bold]{output}[/bold] "
        f"({manifest.train_examples} train, {manifest.validation_examples} validation)"
    )


@train_app.command("agent-sft")
def train_agent_sft_command(
    dataset: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Frozen Agent SFT dataset directory."),
    ],
    hardware: Annotated[str, typer.Option(help="Hardware profile ID.")] = (
        "rtx_4070_ti_super_16gb"
    ),
    model: Annotated[str, typer.Option(help="Model profile ID.")] = "qwen3_0_6b_base",
    experiment: Annotated[str, typer.Option(help="Agent SFT experiment profile ID.")] = "agent_sft",
    artifacts_root: Annotated[Path, typer.Option(help="Parent directory for the run.")] = Path(
        "artifacts/runs"
    ),
    run_id: Annotated[str | None, typer.Option(help="Optional path-safe run ID.")] = None,
    local_files_only: Annotated[bool, typer.Option()] = False,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda.")] = "auto",
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Repeatable scalar override within the Agent SFT profile."),
    ] = None,
    config_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train a LoRA policy directly from stored exact-token agent actions."""

    try:
        resolved = _resolve_agent_sft(
            hardware=hardware,
            model=model,
            experiment=experiment,
            config_dir=config_dir,
            overrides=tuple(set_values or ()),
        )
        context = execute_agent_sft_run(
            resolved,
            dataset_dir=dataset,
            artifacts_root=artifacts_root,
            run_id=run_id,
            local_files_only=local_files_only,
            device=device,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]Agent SFT failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Agent SFT completed: [bold]{context.run_dir}[/bold]")


@train_app.command("agent-rl")
def train_agent_rl_command(
    agent_sft_adapter: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Frozen parent Agent SFT adapter directory.",
        ),
    ],
    tasks_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="MiniSWE suite root."),
    ] = Path("tasks/mini_swe_v1"),
    hardware: Annotated[str, typer.Option(help="Hardware profile ID.")] = (
        "rtx_4070_ti_super_16gb"
    ),
    model: Annotated[str, typer.Option(help="Model profile ID.")] = "qwen3_0_6b_base",
    experiment: Annotated[str, typer.Option(help="Agent RL experiment profile ID.")] = "agent_rl",
    artifacts_root: Annotated[Path, typer.Option(help="Parent directory for the run.")] = Path(
        "artifacts/runs"
    ),
    run_id: Annotated[str | None, typer.Option(help="Optional path-safe run ID.")] = None,
    iteration_limit: Annotated[
        int | None,
        typer.Option(help="Explicit non-representative iteration cap for calibration."),
    ] = None,
    local_files_only: Annotated[bool, typer.Option()] = False,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda.")] = "auto",
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Repeatable scalar override within the Agent RL profile."),
    ] = None,
    config_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Run fresh grouped MiniSWE rollouts and exact-token Agent GRPO updates."""

    try:
        resolved = _resolve_agent_rl(
            hardware=hardware,
            model=model,
            experiment=experiment,
            config_dir=config_dir,
            overrides=tuple(set_values or ()),
        )
        context = execute_agent_rl_run(
            resolved,
            tasks_root=tasks_root,
            agent_sft_adapter_path=agent_sft_adapter,
            artifacts_root=artifacts_root,
            run_id=run_id,
            local_files_only=local_files_only,
            device=device,
            iteration_limit=iteration_limit,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]Agent RL failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Agent RL completed: [bold]{context.run_dir}[/bold]")


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
            selected = move_model(loaded, device)
            console.print(
                f"Loaded {resolved.config.model.source.model_id} on {selected}; "
                f"{loaded.parameters.total:,} parameters; revision {loaded.model_revision}"
            )
            return
        if tasks is None:
            raise ValueError("--tasks is required for --mode eval")
        context = execute_evaluation_run(
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
