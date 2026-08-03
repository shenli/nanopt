"""Strict parsing, exact verification, and checkpoint-agnostic evaluation."""

from nanopt.eval.io import read_arithmetic_tasks
from nanopt.eval.metrics import aggregate_results, pass_at_k, pass_at_k_by_task, wilson_interval
from nanopt.eval.parser import ParseResult, parse_answer
from nanopt.eval.records import EvaluationResult
from nanopt.eval.runner import EvaluationIdentity, EvaluationPlan, evaluate_to_artifacts
from nanopt.eval.verifier import VerificationResult, verify_task_response

__all__ = [
    "EvaluationIdentity",
    "EvaluationPlan",
    "EvaluationResult",
    "ParseResult",
    "VerificationResult",
    "aggregate_results",
    "evaluate_to_artifacts",
    "parse_answer",
    "pass_at_k",
    "pass_at_k_by_task",
    "read_arithmetic_tasks",
    "verify_task_response",
    "wilson_interval",
]
