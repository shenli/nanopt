from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nanopt.config.loader import ConfigError, ConfigRepository
from nanopt.config.provenance import serialize_provenance
from nanopt.config.resolver import parse_override, resolve_config
from nanopt.runtime.artifacts import write_yaml


def test_cli_override_has_highest_precedence_and_provenance(
    config_repository: ConfigRepository,
) -> None:
    result = resolve_config(
        repository=config_repository,
        hardware_id="rtx_4070_ti_super_16gb",
        model_id="qwen3_0_6b_base",
        experiment_id="math_grpo",
        overrides=("rollout.group_size=2", "model.loading.low_cpu_mem_usage=false"),
    )
    assert result.config.experiment.rollout.group_size == 2
    assert result.config.model.loading.low_cpu_mem_usage is False
    assert result.provenance["experiment.rollout.group_size"].source == "cli_override"
    assert result.provenance["model.loading.low_cpu_mem_usage"].source_path.endswith("false")
    assert result.provenance["hardware.memory_budget.hard_peak_reserved_gib"].source == (
        "hardware:rtx_4070_ti_super_16gb"
    )


def test_override_unknown_path_is_rejected(config_repository: ConfigRepository) -> None:
    with pytest.raises(ConfigError, match="unknown override path"):
        resolve_config(
            repository=config_repository,
            hardware_id="rtx_4070_ti_super_16gb",
            model_id="qwen3_0_6b_base",
            experiment_id="math_grpo",
            overrides=("rollout.unknown=2",),
        )


def test_override_type_mismatch_is_rejected(config_repository: ConfigRepository) -> None:
    with pytest.raises(ConfigError, match="resolved configuration is invalid"):
        resolve_config(
            repository=config_repository,
            hardware_id="rtx_4070_ti_super_16gb",
            model_id="qwen3_0_6b_base",
            experiment_id="math_grpo",
            overrides=('rollout.group_size="two"',),
        )


@pytest.mark.parametrize("expression", ["rollout.group_size=[2]", "reward.components={x: 1}"])
def test_override_collections_are_rejected(expression: str) -> None:
    with pytest.raises(ConfigError, match="scalar values only"):
        parse_override(expression)


def test_recipe_selects_profiles_and_stage(config_repository: ConfigRepository) -> None:
    result = resolve_config(
        repository=config_repository,
        recipe_id="math_pipeline",
        recipe_stage_id="grpo",
    )
    assert result.config.hardware.id == "rtx_4070_ti_super_16gb"
    assert result.config.model.id == "qwen3_0_6b_base"
    assert result.config.experiment.id == "math_grpo"
    assert result.config.recipe_stage == "grpo"


def test_resolution_serialization_is_stable(
    tmp_path: Path, config_repository: ConfigRepository
) -> None:
    result = resolve_config(
        repository=config_repository,
        hardware_id="rtx_4070_ti_super_16gb",
        model_id="qwen3_0_6b_base",
        experiment_id="base_eval",
    )
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    value = result.config.model_dump(mode="json", exclude_none=False)
    write_yaml(first, value)
    parsed = yaml.safe_load(first.read_text())
    write_yaml(second, parsed)
    assert first.read_bytes() == second.read_bytes()
    serialized = serialize_provenance(result.provenance)
    assert list(serialized) == sorted(serialized)
