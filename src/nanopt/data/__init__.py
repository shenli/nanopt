"""Deterministic synthetic arithmetic tasks, fingerprints, and split construction."""

from nanopt.data.arithmetic import (
    ArithmeticGeneratorConfig,
    evaluate_ast,
    generate_task,
    generate_tasks,
    render_expression,
    render_trusted_completion,
)
from nanopt.data.fingerprints import canonical_task_hash, dataset_fingerprint
from nanopt.data.schemas import ArithmeticAst, ArithmeticTask, SplitManifest
from nanopt.data.splits import build_splits

__all__ = [
    "ArithmeticAst",
    "ArithmeticGeneratorConfig",
    "ArithmeticTask",
    "SplitManifest",
    "build_splits",
    "canonical_task_hash",
    "dataset_fingerprint",
    "evaluate_ast",
    "generate_task",
    "generate_tasks",
    "render_expression",
    "render_trusted_completion",
]
