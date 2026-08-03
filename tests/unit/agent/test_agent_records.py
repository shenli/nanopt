from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nanopt.agent.records import AgentTaskCard, parse_action


def test_structured_action_accepts_only_allowlisted_tools() -> None:
    action = parse_action(json.dumps({"tool": "read_file", "arguments": {"path": "src/a.py"}}))
    assert action.tool == "read_file"

    with pytest.raises(ValidationError):
        parse_action(json.dumps({"tool": "shell", "arguments": {"command": "id"}}))


@pytest.mark.parametrize(
    "response", ["not json", "[]", '{"tool":"run_tests","arguments":{"command":"id"}}']
)
def test_structured_action_rejects_malformed_or_injected_commands(response: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        parse_action(response)


def test_task_card_rejects_unrecorded_metadata() -> None:
    value = {
        "id": "task",
        "version": "1",
        "split": "smoke",
        "issue": "Fix it",
        "snapshot_sha256": "a" * 64,
        "editable_globs": ["src/*.py"],
        "protected_globs": ["tests/**"],
        "public_test_command": ["python", "-m", "unittest"],
        "hidden_test_command": ["python", "-m", "unittest"],
        "public_tests_total": 1,
        "hidden_tests_total": 1,
        "budgets": {"tool_calls": 2, "test_runs": 1, "turns": 2, "wall_clock_seconds": 10},
        "license": "Apache-2.0",
        "expected_solution": "secret",
    }
    with pytest.raises(ValidationError):
        AgentTaskCard.model_validate(value)
