from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanopt.runtime.artifacts import append_jsonl, read_jsonl, write_json, write_yaml


def test_atomic_document_writers_replace_complete_documents(tmp_path: Path) -> None:
    json_path = tmp_path / "state.json"
    yaml_path = tmp_path / "state.yaml"
    write_json(json_path, {"generation": 1})
    write_json(json_path, {"generation": 2})
    write_yaml(yaml_path, {"z": 1, "a": 2})
    assert json.loads(json_path.read_text()) == {"generation": 2}
    assert yaml_path.read_text().startswith("a: 2\n")
    assert not list(tmp_path.glob(".state.*"))


def test_jsonl_append_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, {"step": 1, "loss": 2.0})
    append_jsonl(path, {"step": 2, "loss": 1.0})
    assert read_jsonl(path) == [
        {"loss": 2.0, "step": 1},
        {"loss": 1.0, "step": 2},
    ]


def test_jsonl_reader_identifies_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok":true}\n{"partial"')
    with pytest.raises(ValueError, match=r"events.jsonl:2"):
        read_jsonl(path)
