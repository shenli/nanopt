"""Deterministic profile composition, overrides, and provenance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import yaml
from pydantic import ValidationError

from nanopt.config.loader import ConfigError, ConfigRepository
from nanopt.config.models import ResolvedConfig, Scalar
from nanopt.config.provenance import (
    ProvenanceEntry,
    ProvenanceMap,
    record_leaves,
)


@dataclass(frozen=True)
class ResolutionResult:
    """A validated resolved configuration and its leaf provenance."""

    config: ResolvedConfig
    provenance: ProvenanceMap
    cli_overrides: tuple[str, ...]


def parse_override(expression: str) -> tuple[str, Scalar]:
    """Parse ``path=value`` while accepting YAML scalar syntax only."""

    if "=" not in expression:
        raise ConfigError(f"override must use path=value syntax: {expression!r}")
    path, raw_value = expression.split("=", 1)
    if not path or any(not component for component in path.split(".")):
        raise ConfigError(f"override path is invalid: {path!r}")
    value = yaml.safe_load(raw_value)
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ConfigError(
            "CLI overrides support scalar values only; lists and mappings are rejected"
        )
    return path, value


def _apply_scalar_override(target: dict[str, Any], path: str, value: Scalar) -> None:
    components = path.split(".")
    cursor: dict[str, Any] = target
    for component in components[:-1]:
        existing = cursor.get(component)
        if not isinstance(existing, dict):
            raise ConfigError(f"unknown or non-mapping override path: {path}")
        cursor = cast(dict[str, Any], existing)
    leaf = components[-1]
    if leaf not in cursor:
        raise ConfigError(f"unknown override path: {path}")
    if isinstance(cursor[leaf], (dict, list)):
        raise ConfigError(f"override path must target a scalar field: {path}")
    cursor[leaf] = value


def resolve_config(
    *,
    repository: ConfigRepository | None = None,
    hardware_id: str | None = None,
    model_id: str | None = None,
    experiment_id: str | None = None,
    recipe_id: str | None = None,
    recipe_stage_id: str | None = None,
    overrides: tuple[str, ...] = (),
) -> ResolutionResult:
    """Resolve named profiles and explicit dotted scalar overrides.

    Profile namespaces stay separate to prevent accidental collisions between fields
    such as model adapters and experiment adapters. Unprefixed overrides target the
    experiment namespace, matching the documented CLI examples.
    """

    repo = repository or ConfigRepository()
    recipe = repo.recipe(recipe_id) if recipe_id else None
    if recipe is not None:
        hardware_id = hardware_id or recipe.hardware
        model_id = model_id or recipe.model

    if not hardware_id or not model_id:
        raise ConfigError("hardware and model profiles are required")

    recipe_stage = None
    if recipe_stage_id:
        if recipe is None:
            raise ConfigError("--stage requires --recipe")
        recipe_stage = next((stage for stage in recipe.stages if stage.id == recipe_stage_id), None)
        if recipe_stage is None:
            raise ConfigError(f"recipe {recipe.id!r} has no stage {recipe_stage_id!r}")
        experiment_id = experiment_id or recipe_stage.experiment

    if not experiment_id:
        raise ConfigError("an experiment profile is required (directly or through a recipe stage)")

    hardware = repo.hardware(hardware_id)
    model = repo.model(model_id)
    experiment = repo.experiment(experiment_id)

    raw: dict[str, Any] = {
        "schema_version": 1,
        "hardware": hardware.model_dump(mode="python"),
        "model": model.model_dump(mode="python"),
        "experiment": experiment.model_dump(mode="python"),
        "recipe": recipe.model_dump(mode="python") if recipe else None,
        "recipe_stage": recipe_stage_id,
    }
    provenance: ProvenanceMap = {
        "schema_version": ProvenanceEntry("package_default", "schema_version")
    }
    provenance.update(
        record_leaves(raw["hardware"], prefix="hardware", source=f"hardware:{hardware.id}")
    )
    provenance.update(record_leaves(raw["model"], prefix="model", source=f"model:{model.id}"))
    provenance.update(
        record_leaves(raw["experiment"], prefix="experiment", source=f"experiment:{experiment.id}")
    )
    if recipe:
        provenance.update(
            record_leaves(raw["recipe"], prefix="recipe", source=f"recipe:{recipe.id}")
        )
    if recipe_stage_id:
        provenance["recipe_stage"] = ProvenanceEntry(
            source=f"recipe:{recipe.id if recipe else ''}", source_path="stages.id"
        )

    if recipe_stage is not None:
        for path, value in recipe_stage.overrides.items():
            _apply_scalar_override(cast(dict[str, Any], raw["experiment"]), path, value)
            provenance[f"experiment.{path}"] = ProvenanceEntry(
                source=f"recipe_stage:{recipe_stage.id}", source_path=f"overrides.{path}"
            )

    for expression in overrides:
        path, value = parse_override(expression)
        if path.startswith("hardware."):
            namespace, relative = "hardware", path.removeprefix("hardware.")
        elif path.startswith("model."):
            namespace, relative = "model", path.removeprefix("model.")
        elif path.startswith("experiment."):
            namespace, relative = "experiment", path.removeprefix("experiment.")
        else:
            namespace, relative = "experiment", path
        _apply_scalar_override(cast(dict[str, Any], raw[namespace]), relative, value)
        provenance[f"{namespace}.{relative}"] = ProvenanceEntry(
            source="cli_override", source_path=expression
        )

    try:
        resolved = ResolvedConfig.model_validate(deepcopy(raw))
    except ValidationError as exc:
        raise ConfigError(f"resolved configuration is invalid:\n{exc}") from exc
    return ResolutionResult(resolved, provenance, overrides)
