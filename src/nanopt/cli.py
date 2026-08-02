"""NanoPT command-line interface for the repository foundation milestone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from nanopt.config.loader import ConfigError, ConfigRepository
from nanopt.config.provenance import serialize_provenance
from nanopt.config.resolver import resolve_config
from nanopt.runtime.artifacts import write_json, write_yaml
from nanopt.runtime.doctor import DoctorReport, collect_doctor_report
from nanopt.version import __version__

app = typer.Typer(
    name="nanopt",
    help="NanoPT: a white-box course and reference implementation for post-training.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(help="Validate and deterministically resolve configuration profiles.")
artifacts_app = typer.Typer(help="Inspect local NanoPT run artifacts.")
app.add_typer(config_app, name="config")
app.add_typer(artifacts_app, name="artifacts")
console = Console()


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
    """Expose only implemented, inspectable M1 commands."""


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
