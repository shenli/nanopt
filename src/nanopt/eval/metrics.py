"""Small, labeled evaluation estimators with hand-checkable implementations."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist

from nanopt.eval.records import EvaluationResult


@dataclass(frozen=True)
class BinomialInterval:
    """A confidence interval for a rate, including the sample count used."""

    estimate: float
    lower: float
    upper: float
    count: int
    successes: int
    method: str = "wilson"


def pass_at_k(*, samples: int, correct: int, k: int) -> float:
    r"""Compute the standard unbiased pass@k estimator without large combinations.

    The estimator is

    ``1 - C(samples - correct, k) / C(samples, k)``.

    The ratio is evaluated as a product of at most ``k`` fractions, avoiding enormous integer
    combinations. If fewer than ``k`` failures exist, at least one of any ``k`` samples must pass,
    so the result is exactly one.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if correct < 0 or correct > samples:
        raise ValueError("correct must be between zero and samples")
    if k <= 0 or k > samples:
        raise ValueError("k must be between one and samples")
    failures = samples - correct
    if failures < k:
        return 1.0
    all_fail_probability = 1.0
    for offset in range(k):
        all_fail_probability *= (failures - offset) / (samples - offset)
    return 1.0 - all_fail_probability


def wilson_interval(successes: int, count: int, *, confidence: float = 0.95) -> BinomialInterval:
    """Return a two-sided Wilson score interval for an accuracy-like rate.

    Wilson intervals stay within ``[0, 1]`` and behave sensibly for all-success and all-failure
    fixtures, unlike the elementary normal approximation.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if successes < 0 or successes > count:
        raise ValueError("successes must be between zero and count")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    rate = successes / count
    z2 = z * z
    denominator = 1 + z2 / count
    center = (rate + z2 / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z2 / (4 * count * count)) / denominator
    return BinomialInterval(
        estimate=rate,
        lower=max(0.0, center - radius),
        upper=min(1.0, center + radius),
        count=count,
        successes=successes,
    )


def pass_at_k_by_task(results: list[EvaluationResult], *, k: int) -> BinomialInterval:
    """Average per-task pass@k estimates and report a Wilson interval over direct task success.

    Every task must have the same number of samples and at least ``k`` samples. The headline
    estimate is the standard per-task estimator. The interval uses the directly observed event
    “at least one of the first k samples passed,” labeled in report metadata to avoid conflating the
    two quantities.
    """

    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        grouped[result.task_id].append(result)
    if not grouped:
        raise ValueError("at least one result is required")
    sample_counts = {len(values) for values in grouped.values()}
    if len(sample_counts) != 1:
        raise ValueError("every task must have the same number of samples")
    samples = sample_counts.pop()
    if samples < k:
        raise ValueError(f"every task needs at least {k} samples")
    estimates: list[float] = []
    direct_successes = 0
    for values in grouped.values():
        ordered = sorted(values, key=lambda item: item.sample_index)
        if len({item.sample_index for item in ordered}) != len(ordered):
            raise ValueError("sample indexes must be unique within every task")
        correct = sum(item.verifier_status == "correct" for item in ordered)
        estimates.append(pass_at_k(samples=samples, correct=correct, k=k))
        direct_successes += any(item.verifier_status == "correct" for item in ordered[:k])
    interval = wilson_interval(direct_successes, len(grouped))
    return BinomialInterval(
        estimate=sum(estimates) / len(estimates),
        lower=interval.lower,
        upper=interval.upper,
        count=interval.count,
        successes=interval.successes,
        method=f"pass@{k}-estimator; direct-first-{k}-wilson-interval",
    )


def aggregate_results(results: list[EvaluationResult]) -> dict[str, object]:
    """Aggregate correctness, parsing, termination, and length without discarding examples."""

    if not results:
        raise ValueError("at least one result is required")
    correct = sum(item.verifier_status == "correct" for item in results)
    parsed = sum(item.parser_status == "valid" for item in results)
    eos = sum(item.finish_reason == "eos" for item in results)
    lengths = [len(item.completion_token_ids or []) for item in results]
    accuracy = wilson_interval(correct, len(results))
    parse_rate = wilson_interval(parsed, len(results))
    return {
        "schema_version": 1,
        "examples": len(results),
        "tasks": len({item.task_id for item in results}),
        "accuracy": accuracy.__dict__,
        "parse_rate": parse_rate.__dict__,
        "eos_fraction": eos / len(results),
        "completion_tokens": {
            "mean": sum(lengths) / len(lengths),
            "minimum": min(lengths),
            "maximum": max(lengths),
        },
    }
