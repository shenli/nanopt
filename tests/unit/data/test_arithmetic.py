"""Property-style arithmetic generator tests over many deterministic seeds."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml
from pydantic import ValidationError

from nanopt.data.arithmetic import (
    ArithmeticGeneratorConfig,
    evaluate_ast,
    generate_task,
    generate_tasks,
    render_expression,
    render_trusted_completion,
)

FAMILIES = (
    "addition_subtraction",
    "multiplication",
    "exact_division",
    "mixed_precedence",
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", range(20))
def test_generated_target_matches_exact_ast_evaluation(family: Any, seed: int) -> None:
    task = generate_task(family=family, difficulty=1 + seed % 3, seed=seed)
    value = evaluate_ast(task.canonical_ast)

    assert isinstance(value, Fraction)
    assert task.target.canonical_answer == str(value.numerator)
    assert task.target.answer_type == "integer"
    assert task.prompt.startswith(f"Compute {render_expression(task.canonical_ast)}.")
    assert render_trusted_completion(task).endswith(
        f"<answer>{task.target.canonical_answer}</answer>"
    )


def test_exact_division_generator_never_divides_by_zero() -> None:
    for seed in range(100):
        task = generate_task(family="exact_division", difficulty=2, seed=seed)
        assert task.canonical_ast.right is not None
        assert evaluate_ast(task.canonical_ast.right) != 0
        assert evaluate_ast(task.canonical_ast).denominator == 1


def test_collection_generation_is_reproducible_and_unique() -> None:
    config = ArithmeticGeneratorConfig(seed=42, count=40)

    first = generate_tasks(config)
    second = generate_tasks(config)

    assert [task.model_dump() for task in first] == [task.model_dump() for task in second]
    assert len({task.task_id for task in first}) == 40
    assert {task.family for task in first} == set(FAMILIES)


def test_checked_in_generator_config_and_tasks_validate_against_schemas(
    project_root: Path,
) -> None:
    config_value = yaml.safe_load(
        (project_root / "tasks/arithmetic/generator_config.yaml").read_text()
    )
    config = ArithmeticGeneratorConfig.model_validate(config_value)
    task_schema = json.loads((project_root / "specs/schemas/task.schema.json").read_text())

    assert config.count == 128
    for task in generate_tasks(config)[:10]:
        jsonschema.validate(task.model_dump(mode="json", exclude_none=True), task_schema)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"families": []}, "must not be empty"),
        ({"families": ["multiplication", "multiplication"]}, "duplicates"),
        ({"minimum_operand": 2, "maximum_operand": 2}, "must be smaller"),
    ],
)
def test_generator_config_rejects_invalid_ranges(
    updates: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((ValidationError, ValueError), match=message):
        ArithmeticGeneratorConfig(seed=1, count=2, **updates)


def test_generate_task_rejects_invalid_difficulty_and_range() -> None:
    with pytest.raises(ValueError, match="difficulty"):
        generate_task(family="multiplication", difficulty=0, seed=1)
    with pytest.raises(ValueError, match="must be smaller"):
        generate_task(
            family="multiplication",
            difficulty=1,
            seed=1,
            minimum_operand=2,
            maximum_operand=2,
        )
