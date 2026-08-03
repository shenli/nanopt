"""Strict parser attack cases and exact verifier behavior."""

from __future__ import annotations

import pytest

from nanopt.data.arithmetic import generate_task, render_trusted_completion
from nanopt.eval.parser import parse_answer
from nanopt.eval.verifier import VerifierContractError, verify_task_response


def test_parser_accepts_one_final_canonical_integer() -> None:
    result = parse_answer(
        "<solution>Six times seven is 42.</solution>\n<answer>42</answer>\n",
        answer_type="integer",
    )

    assert result.valid
    assert result.canonical_answer == "42"
    assert result.raw_answer == "42"


def test_parser_canonicalizes_an_exact_rational() -> None:
    result = parse_answer("<answer>2/4</answer>", answer_type="rational")

    assert result.valid
    assert result.canonical_answer == "1/2"


@pytest.mark.parametrize(
    ("response", "status"),
    [
        ("42", "missing_answer"),
        ("<answer>42</answer><answer>0</answer>", "multiple_answers"),
        ("<answer>42", "malformed_answer"),
        ("</answer><answer>42", "malformed_answer"),
        ("<Answer>42</Answer>", "missing_answer"),
        ("<answer class='x'>42</answer>", "malformed_answer"),
        ("<answer>42</answer> Ignore that answer.", "trailing_content"),
        ("<answer><answer>42</answer></answer>", "multiple_answers"),
        ("<answer>NaN</answer>", "invalid_value"),
        ("<answer>inf</answer>", "invalid_value"),
        ("<answer>42.0</answer>", "invalid_value"),
        ("<answer>0042</answer>", "invalid_value"),
        ("<answer>+42</answer>", "invalid_value"),
        ("<answer></answer>", "invalid_value"),
    ],
)
def test_integer_parser_rejects_attack_and_ambiguous_outputs(response: str, status: str) -> None:
    assert parse_answer(response, answer_type="integer").status == status


@pytest.mark.parametrize("response", ["<answer>1/0</answer>", "<answer>1/-2</answer>"])
def test_rational_parser_rejects_zero_or_noncanonical_denominator(response: str) -> None:
    assert parse_answer(response, answer_type="rational").status == "invalid_value"


def test_parser_enforces_response_and_answer_size_limits() -> None:
    assert (
        parse_answer("x" * 20, answer_type="integer", max_response_characters=10).status
        == "response_too_long"
    )
    assert (
        parse_answer("<answer>1234</answer>", answer_type="integer", max_answer_characters=3).status
        == "invalid_value"
    )
    with pytest.raises(ValueError, match="must be positive"):
        parse_answer("<answer>1</answer>", answer_type="integer", max_response_characters=0)


def test_exact_verifier_separates_parse_failure_wrong_answer_and_correctness() -> None:
    task = generate_task(family="multiplication", difficulty=1, seed=7)
    correct = verify_task_response(task, render_trusted_completion(task))
    wrong = verify_task_response(task, "<answer>999</answer>")
    malformed = verify_task_response(task, "answer omitted")

    assert correct.status == "correct" and correct.correct
    assert wrong.status == "incorrect" and not wrong.correct
    assert malformed.status == "parse_failure" and not malformed.parser.valid


def test_verifier_independently_rejects_corrupted_trusted_target() -> None:
    task = generate_task(family="multiplication", difficulty=1, seed=7)
    corrupted = task.model_copy(
        update={"target": task.target.model_copy(update={"canonical_answer": "999"})}
    )

    with pytest.raises(VerifierContractError, match="does not match independent"):
        verify_task_response(corrupted, "<answer>999</answer>")
