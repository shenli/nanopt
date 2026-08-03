from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml


def test_all_json_schemas_are_valid(project_root: Path) -> None:
    for path in sorted((project_root / "specs/schemas").glob("*.json")):
        schema = json.loads(path.read_text())
        jsonschema.validators.validator_for(schema).check_schema(schema)


def test_all_public_yaml_profiles_are_mappings(project_root: Path) -> None:
    for path in sorted((project_root / "configs").rglob("*.yaml")):
        assert isinstance(yaml.safe_load(path.read_text()), dict), path


def test_all_agent_task_cards_match_public_schema(project_root: Path) -> None:
    schema = json.loads((project_root / "specs/schemas/agent_task.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for path in sorted((project_root / "tasks/mini_swe_v1").glob("*/task.yaml")):
        validator.validate(yaml.safe_load(path.read_text()))


def test_release_manifest_matches_public_schema(project_root: Path) -> None:
    schema = json.loads((project_root / "specs/schemas/release.schema.json").read_text())
    release = yaml.safe_load(
        (project_root / "configs/releases/v0_1_0.yaml").read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        release
    )
