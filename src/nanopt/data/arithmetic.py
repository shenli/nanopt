"""Safe deterministic arithmetic-AST generation and trusted rendering."""

from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction
from typing import Literal

from pydantic import Field

from nanopt.data.schemas import (
    AnswerType,
    ArithmeticAst,
    ArithmeticTask,
    DataModel,
    TaskProvenance,
    TaskTarget,
)

TaskFamily = Literal[
    "addition_subtraction",
    "multiplication",
    "exact_division",
    "mixed_precedence",
]

GENERATOR_VERSION: Literal["arithmetic-v1"] = "arithmetic-v1"


def _default_families() -> list[TaskFamily]:
    return [
        "addition_subtraction",
        "multiplication",
        "exact_division",
        "mixed_precedence",
    ]


class ArithmeticGeneratorConfig(DataModel):
    """Inputs that fully determine a generated task collection."""

    schema_version: Literal[1] = 1
    generator_version: Literal["arithmetic-v1"] = GENERATOR_VERSION
    seed: int
    count: int = Field(gt=0)
    families: list[TaskFamily] = Field(default_factory=_default_families)
    minimum_operand: int = -20
    maximum_operand: int = 20

    def model_post_init(self, _context: object) -> None:
        if not self.families:
            raise ValueError("families must not be empty")
        if len(set(self.families)) != len(self.families):
            raise ValueError("families must not contain duplicates")
        if self.minimum_operand >= self.maximum_operand:
            raise ValueError("minimum_operand must be smaller than maximum_operand")


def integer(value: int) -> ArithmeticAst:
    """Create one integer literal node."""

    return ArithmeticAst(kind="integer", value=value)


def binary(op: str, left: ArithmeticAst, right: ArithmeticAst) -> ArithmeticAst:
    """Create one validated binary node."""

    return ArithmeticAst.model_validate(
        {"kind": "binary", "op": op, "left": left, "right": right},
        strict=True,
    )


def evaluate_ast(node: ArithmeticAst) -> Fraction:
    """Evaluate a trusted AST exactly with ``Fraction`` and never Python ``eval``."""

    if node.kind == "integer":
        if node.value is None:  # Protected by the model validator; keeps narrowing explicit.
            raise ValueError("integer AST is missing value")
        return Fraction(node.value)
    if node.left is None or node.right is None or node.op is None:
        raise ValueError("binary AST is incomplete")
    left = evaluate_ast(node.left)
    right = evaluate_ast(node.right)
    if node.op == "add":
        return left + right
    if node.op == "subtract":
        return left - right
    if node.op == "multiply":
        return left * right
    if right == 0:
        raise ZeroDivisionError("arithmetic AST divides by zero")
    return left / right


def render_expression(node: ArithmeticAst) -> str:
    """Render a fully parenthesized expression whose structure is unambiguous."""

    if node.kind == "integer":
        if node.value is None:
            raise ValueError("integer AST is missing value")
        return str(node.value)
    if node.left is None or node.right is None or node.op is None:
        raise ValueError("binary AST is incomplete")
    symbols = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
    return f"({render_expression(node.left)} {symbols[node.op]} {render_expression(node.right)})"


def canonical_fraction(value: Fraction) -> tuple[AnswerType, str]:
    """Return answer type and canonical text for one exact rational value."""

    if value.denominator == 1:
        return "integer", str(value.numerator)
    return "rational", f"{value.numerator}/{value.denominator}"


def _ast_for_family(
    family: TaskFamily,
    *,
    difficulty: int,
    rng: random.Random,
    minimum_operand: int,
    maximum_operand: int,
) -> ArithmeticAst:
    def operand() -> int:
        return rng.randint(minimum_operand, maximum_operand)

    if family == "addition_subtraction":
        op = rng.choice(("add", "subtract"))
        return binary(op, integer(operand()), integer(operand()))
    if family == "multiplication":
        limit = max(2, min(12 + difficulty * 2, max(abs(minimum_operand), maximum_operand)))
        return binary(
            "multiply", integer(rng.randint(-limit, limit)), integer(rng.randint(-limit, limit))
        )
    if family == "exact_division":
        divisor = 0
        while divisor == 0:
            divisor = operand()
        quotient = operand()
        return binary("divide", integer(divisor * quotient), integer(divisor))

    first_op = rng.choice(("add", "subtract"))
    second_op = rng.choice(("multiply", "add", "subtract"))
    left = binary(first_op, integer(operand()), integer(operand()))
    return binary(second_op, left, integer(operand()))


def generate_task(
    *,
    family: TaskFamily,
    difficulty: int,
    seed: int,
    minimum_operand: int = -20,
    maximum_operand: int = 20,
) -> ArithmeticTask:
    """Generate one task reproducibly from its family, difficulty, and task seed."""

    if difficulty < 1 or difficulty > 5:
        raise ValueError("difficulty must be between 1 and 5")
    if minimum_operand >= maximum_operand:
        raise ValueError("minimum_operand must be smaller than maximum_operand")
    rng = random.Random(seed)
    ast = _ast_for_family(
        family,
        difficulty=difficulty,
        rng=rng,
        minimum_operand=minimum_operand,
        maximum_operand=maximum_operand,
    )
    answer_type, canonical_answer = canonical_fraction(evaluate_ast(ast))
    ast_value = ast.model_dump(mode="json", exclude_none=True)
    identity = json.dumps(
        {"family": family, "difficulty": difficulty, "ast": ast_value},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    task_id = f"arith_{hashlib.sha256(identity).hexdigest()[:16]}"
    expression = render_expression(ast)
    prompt = (
        f"Compute {expression}. Return a concise derivation inside <solution> tags and the exact "
        "final value inside one <answer> tag."
    )
    return ArithmeticTask(
        task_id=task_id,
        family=family,
        difficulty=difficulty,
        prompt=prompt,
        canonical_ast=ast,
        target=TaskTarget(answer_type=answer_type, canonical_answer=canonical_answer),
        provenance=TaskProvenance(generator_version=GENERATOR_VERSION, seed=seed),
    )


def generate_tasks(config: ArithmeticGeneratorConfig) -> list[ArithmeticTask]:
    """Generate a unique deterministic collection from one master configuration."""

    rng = random.Random(config.seed)
    tasks: list[ArithmeticTask] = []
    seen_ids: set[str] = set()
    attempts = 0
    maximum_attempts = config.count * 100
    while len(tasks) < config.count and attempts < maximum_attempts:
        attempts += 1
        family = config.families[len(tasks) % len(config.families)]
        difficulty = 1 + (len(tasks) % 3)
        task_seed = rng.randrange(0, 2**63)
        task = generate_task(
            family=family,
            difficulty=difficulty,
            seed=task_seed,
            minimum_operand=config.minimum_operand,
            maximum_operand=config.maximum_operand,
        )
        if task.task_id not in seen_ids:
            seen_ids.add(task.task_id)
            tasks.append(task)
    if len(tasks) != config.count:
        raise RuntimeError(
            f"could generate only {len(tasks)} unique tasks after {maximum_attempts} attempts"
        )
    return tasks


def render_trusted_completion(task: ArithmeticTask) -> str:
    """Render a deterministic trusted solution directly from the AST and exact target."""

    expression = render_expression(task.canonical_ast)
    answer = canonical_fraction(evaluate_ast(task.canonical_ast))[1]
    return f"<solution>{expression} = {answer}.</solution>\n<answer>{answer}</answer>"
