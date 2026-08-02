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
