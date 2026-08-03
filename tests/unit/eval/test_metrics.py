from __future__ import annotations

import pytest

from nanopt.eval.metrics import aggregate_results, pass_at_k, pass_at_k_by_task, wilson_interval
from nanopt.eval.records import EvaluationResult


def _result(task: str, sample: int, *, correct: bool, parsed: bool = True) -> EvaluationResult:
    return EvaluationResult(
        result_id=f"{task}-{sample}",
        run_id="run",
        checkpoint_id="base",
        task_id=task,
        split="test_iid",
        sample_index=sample,
        seed=sample,
        generation_config_sha256="a" * 64,
        prompt_token_ids=[1],
        completion_token_ids=[2, 3],
        response_text="<answer>1</answer>",
        parser_status="valid" if parsed else "invalid",
        parsed_answer="1" if parsed else None,
        verifier_status="correct" if correct else "incorrect",
        reward_components={"correctness": float(correct)},
        finish_reason="eos",
        generation_seconds=0.1,
    )


def test_pass_at_k_matches_hand_computed_fixture() -> None:
    # 4 samples, 2 correct, k=2: 1 - C(2, 2) / C(4, 2) = 5/6.
    assert pass_at_k(samples=4, correct=2, k=2) == pytest.approx(5 / 6)
    assert pass_at_k(samples=4, correct=3, k=2) == 1.0
    assert pass_at_k(samples=4, correct=0, k=2) == 0.0


def test_pass_at_k_validates_counts() -> None:
    for arguments in (
        {"samples": 0, "correct": 0, "k": 1},
        {"samples": 2, "correct": 3, "k": 1},
        {"samples": 2, "correct": 1, "k": 0},
        {"samples": 2, "correct": 1, "k": 3},
    ):
        with pytest.raises(ValueError):
            pass_at_k(**arguments)


def test_wilson_interval_is_bounded_for_edge_rates() -> None:
    none = wilson_interval(0, 10)
    all_correct = wilson_interval(10, 10)
    assert none.estimate == 0
    assert 0 <= none.lower <= none.upper < 1
    assert all_correct.estimate == 1
    assert 0 < all_correct.lower <= all_correct.upper <= 1


def test_pass_at_k_groups_samples_by_task() -> None:
    results = [
        _result("a", 0, correct=True),
        _result("a", 1, correct=False),
        _result("b", 0, correct=False),
        _result("b", 1, correct=False),
    ]
    metric = pass_at_k_by_task(results, k=2)
    assert metric.estimate == pytest.approx(0.5)
    assert metric.count == 2
    assert metric.successes == 1


def test_aggregate_results_keeps_labeled_counts() -> None:
    summary = aggregate_results(
        [_result("a", 0, correct=True), _result("b", 0, correct=False, parsed=False)]
    )
    assert summary["examples"] == 2
    assert summary["tasks"] == 2
    assert summary["accuracy"]["estimate"] == 0.5  # type: ignore[index]
    assert summary["parse_rate"]["estimate"] == 0.5  # type: ignore[index]


def test_metric_aggregation_rejects_empty_or_unbalanced_groups() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_results([])
    with pytest.raises(ValueError, match="same number"):
        pass_at_k_by_task(
            [
                _result("a", 0, correct=True),
                _result("a", 1, correct=True),
                _result("b", 0, correct=False),
            ],
            k=1,
        )
    duplicate = [_result("a", 0, correct=True), _result("a", 0, correct=False)]
    with pytest.raises(ValueError, match="indexes must be unique"):
        pass_at_k_by_task(duplicate, k=1)
