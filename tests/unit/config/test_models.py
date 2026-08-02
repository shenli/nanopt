from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nanopt.config.loader import ConfigError, ConfigRepository


def test_all_canonical_profiles_validate(config_repository: ConfigRepository) -> None:
    config_repository.hardware("rtx_4070_ti_super_16gb")
    config_repository.model("qwen3_0_6b_base")
    config_repository.model("qwen3_0_6b_instruct")
    for experiment in (
        "base_eval",
        "math_sft",
        "math_dpo",
        "math_grpo",
        "ppo_toy",
        "mini_swe_rollout",
    ):
        config_repository.experiment(experiment)
    config_repository.recipe("math_pipeline")


def test_public_configs_match_handoff_specifications(project_root: Path) -> None:
    for public_path in sorted((project_root / "configs").rglob("*.yaml")):
        relative = public_path.relative_to(project_root / "configs")
        spec_path = project_root / "specs" / "configs" / relative
        assert yaml.safe_load(spec_path.read_text()) == yaml.safe_load(public_path.read_text())


def test_unknown_nested_key_is_rejected(
    tmp_path: Path, project_root: Path, config_repository: ConfigRepository
) -> None:
    raw = yaml.safe_load(
        (project_root / "configs/hardware/rtx_4070_ti_super_16gb.yaml").read_text()
    )
    raw["memory_budget"]["mystery_limit"] = 1
    root = tmp_path / "configs"
    path = root / "hardware" / "bad.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw))
    repository = ConfigRepository(root)
    with pytest.raises(ConfigError, match="mystery_limit"):
        repository.hardware("bad")


@pytest.mark.parametrize("profile_id", ["../secret", "x/y", r"x\\y", ""])
def test_profile_ids_cannot_escape_config_root(
    config_repository: ConfigRepository, profile_id: str
) -> None:
    with pytest.raises(ConfigError, match="invalid profile id"):
        config_repository.hardware(profile_id)


def test_validated_hardware_requires_evidence(tmp_path: Path, project_root: Path) -> None:
    raw = yaml.safe_load(
        (project_root / "configs/hardware/rtx_4070_ti_super_16gb.yaml").read_text()
    )
    raw["id"] = "invalid_validated"
    raw["support_status"] = "validated"
    path = tmp_path / "hardware" / "invalid_validated.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="evidence manifest"):
        ConfigRepository(tmp_path).hardware("invalid_validated")
