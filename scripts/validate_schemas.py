"""Parse every checked-in YAML/JSON file and validate JSON Schema definitions."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml


def main() -> int:
    schemas = sorted(Path("specs/schemas").glob("*.json"))
    yaml_files = sorted(Path("configs").rglob("*.yaml"))
    for path in schemas:
        value = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.validators.validator_for(value).check_schema(value)
    for path in yaml_files:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain a mapping")
    print(f"Validated {len(schemas)} JSON schemas and parsed {len(yaml_files)} YAML profiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
