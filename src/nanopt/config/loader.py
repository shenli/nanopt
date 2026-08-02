"""YAML profile discovery and strict model validation."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import yaml
from pydantic import BaseModel, ValidationError

from nanopt.config.models import (
    ExperimentProfile,
    HardwareProfile,
    ModelProfile,
    RecipeProfile,
    StrictModel,
)


class ConfigError(ValueError):
    """Raised when a configuration profile cannot be loaded or validated."""


class ReadablePath(Protocol):
    """The subset shared by ``Path`` and importlib resource paths."""

    def joinpath(self, *descendants: str) -> ReadablePath: ...

    def is_file(self) -> bool: ...

    def read_text(self, encoding: str = "utf-8") -> str: ...


ProfileModel = TypeVar("ProfileModel", bound=BaseModel)


def default_config_root() -> ReadablePath:
    """Locate checkout profiles first, then profiles bundled in the installed wheel.

    This order lets contributors edit ``configs/`` and see changes immediately while installed
    users receive the exact same profile layout embedded in the wheel.
    """

    checkout_root = Path.cwd() / "configs"
    if checkout_root.is_dir():
        return cast(ReadablePath, checkout_root)

    source_root = Path(__file__).resolve().parents[3] / "configs"
    if source_root.is_dir():
        return cast(ReadablePath, source_root)

    return cast(ReadablePath, resources.files("nanopt").joinpath("_builtin_configs"))


def load_yaml(path: ReadablePath) -> dict[str, Any]:
    """Load a YAML mapping without performing implicit type coercion beyond YAML rules."""

    if not path.is_file():
        raise ConfigError(f"configuration profile does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"configuration profile must contain a string-keyed mapping: {path}")
    return cast(dict[str, Any], value)


def load_model(path: ReadablePath, model: type[ProfileModel]) -> ProfileModel:
    """Load a YAML profile and validate it with an extra-forbidding Pydantic model."""

    try:
        return model.model_validate(load_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration profile {path}:\n{exc}") from exc


class ExperimentEnvelope(StrictModel):
    """Give Pydantic a field on which to apply the experiment stage discriminator."""

    experiment: ExperimentProfile


class ConfigRepository:
    """Read named profiles from the canonical configuration directory layout."""

    def __init__(self, root: ReadablePath | None = None) -> None:
        self.root = root or default_config_root()

    def _path(self, category: str, profile_id: str) -> ReadablePath:
        # Profile IDs become filenames, so reject separators and traversal before joining paths.
        if not profile_id or any(part in profile_id for part in ("/", "\\", "..")):
            raise ConfigError(f"invalid profile id: {profile_id!r}")
        return self.root.joinpath(category, f"{profile_id}.yaml")

    def hardware(self, profile_id: str) -> HardwareProfile:
        return load_model(self._path("hardware", profile_id), HardwareProfile)

    def model(self, profile_id: str) -> ModelProfile:
        return load_model(self._path("models", profile_id), ModelProfile)

    def experiment(self, profile_id: str) -> ExperimentProfile:
        raw = load_yaml(self._path("experiments", profile_id))
        try:
            return ExperimentEnvelope.model_validate({"experiment": raw}).experiment
        except ValidationError as exc:
            raise ConfigError(f"invalid experiment profile {profile_id!r}:\n{exc}") from exc

    def recipe(self, profile_id: str) -> RecipeProfile:
        return load_model(self._path("recipes", profile_id), RecipeProfile)
