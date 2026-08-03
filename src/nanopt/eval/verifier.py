"""Exact-answer verification against independently evaluated trusted AST state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nanopt.data.arithmetic import canonical_fraction, evaluate_ast
from nanopt.data.schemas import ArithmeticTask
from nanopt.eval.parser import ParseResult, parse_answer

VerificationStatus = Literal["correct", "incorrect", "parse_failure"]


class VerifierContractError(ValueError):
    """Raised when checked-in trusted task state contradicts independent AST evaluation."""


@dataclass(frozen=True)
class VerificationResult:
    """Separate parser validity, exact correctness, and trusted/candidate values."""

    status: VerificationStatus
    parser: ParseResult
    correct: bool
    expected_answer: str
    candidate_answer: str | None


def verify_task_response(task: ArithmeticTask, response: str) -> VerificationResult:
    """Re-evaluate the task AST, validate trusted target state, then compare exact answers."""

    expected_type, expected_answer = canonical_fraction(evaluate_ast(task.canonical_ast))
    if task.target.answer_type != expected_type or task.target.canonical_answer != expected_answer:
        raise VerifierContractError(
            "task target does not match independent canonical AST evaluation: "
            f"stored=({task.target.answer_type}, {task.target.canonical_answer!r}), "
            f"evaluated=({expected_type}, {expected_answer!r})"
        )
    parsed = parse_answer(response, answer_type=task.target.answer_type)
    if not parsed.valid:
        return VerificationResult(
            status="parse_failure",
            parser=parsed,
            correct=False,
            expected_answer=expected_answer,
            candidate_answer=None,
        )
    correct = parsed.canonical_answer == expected_answer
    return VerificationResult(
        status="correct" if correct else "incorrect",
        parser=parsed,
        correct=correct,
        expected_answer=expected_answer,
        candidate_answer=parsed.canonical_answer,
    )
