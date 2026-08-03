"""Strict arithmetic RLVR rewards with parser/verifier failure separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nanopt.config.models import RewardComponentsConfig
from nanopt.data.schemas import ArithmeticTask
from nanopt.eval.verifier import VerifierContractError, verify_task_response


@dataclass(frozen=True)
class RewardResult:
    """Scalar reward plus named components and non-conflated failure states."""

    reward: float
    components: dict[str, float]
    parser_status: Literal["valid", "invalid", "error"]
    parsed_answer: str | None
    verifier_status: Literal["correct", "incorrect", "not_run", "error"]


def arithmetic_rlvr_reward(
    task: ArithmeticTask,
    response: str,
    weights: RewardComponentsConfig,
    *,
    completion_tokens: int,
) -> RewardResult:
    """Score one decoded view while trusted correctness remains AST-derived.

    Decoding is permitted for the parser/reward boundary. Training continues to consume the exact
    sampled token IDs and never re-tokenizes this text.
    """

    if completion_tokens <= 0:
        raise ValueError("reward requires at least one completion token")
    try:
        verification = verify_task_response(task, response)
    except VerifierContractError:
        return RewardResult(
            reward=0.0,
            components={
                "parser_valid": 0.0,
                "format_reward": 0.0,
                "correctness_reward": 0.0,
                "length_penalty": 0.0,
            },
            parser_status="error",
            parsed_answer=None,
            verifier_status="error",
        )
    parser_valid = float(verification.parser.valid)
    correctness = float(verification.correct)
    # The v0.1 baseline sets this weight to zero. Keeping the component explicit prevents a later
    # length penalty from being smuggled into an unnamed reward transformation.
    length_penalty = float(completion_tokens)
    components = {
        "parser_valid": parser_valid,
        "format_reward": parser_valid,
        "correctness_reward": correctness,
        "length_penalty": length_penalty,
    }
    reward = (
        weights.correctness * correctness
        + weights.format * parser_valid
        - weights.length_penalty * length_penalty
    )
    return RewardResult(
        reward=reward,
        components=components,
        parser_status="valid" if verification.parser.valid else "invalid",
        parsed_answer=verification.candidate_answer,
        verifier_status=(
            "correct"
            if verification.correct
            else ("incorrect" if verification.parser.valid else "not_run")
        ),
    )


def reward_hacking_suite(
    task: ArithmeticTask, weights: RewardComponentsConfig
) -> list[dict[str, str | float | bool]]:
    """Exercise fixed parser attacks and prove none receives correctness credit."""

    expected = task.target.canonical_answer
    wrong = "0" if expected != "0" else "1"
    attacks = {
        "wrong_answer": f"<answer>{wrong}</answer>",
        "multiple_answers": f"<answer>{expected}</answer><answer>{wrong}</answer>",
        "trailing_content": f"<answer>{expected}</answer> ignore this",
        "case_shifted_tag": f"<Answer>{expected}</Answer>",
        "answer_attribute": f"<answer trusted='yes'>{expected}</answer>",
    }
    results: list[dict[str, str | float | bool]] = []
    for name, response in attacks.items():
        result = arithmetic_rlvr_reward(task, response, weights, completion_tokens=1)
        correctness = result.components["correctness_reward"]
        results.append(
            {
                "case": name,
                "response": response,
                "reward": result.reward,
                "correctness_reward": correctness,
                "passed": correctness == 0.0,
            }
        )
    if not all(bool(result["passed"]) for result in results):
        raise RuntimeError("reward-hacking suite granted correctness credit to an attack")
    return results
