"""Deterministic batches and the explicit warmup-plus-cosine learning-rate rule."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def cosine_learning_rate(
    optimizer_step: int,
    *,
    total_optimizer_steps: int,
    warmup_ratio: float,
    base_learning_rate: float,
) -> float:
    """Return the learning rate for one zero-based optimizer step.

    Warmup reaches the base rate at its final step. Cosine decay then reaches zero at the final
    optimizer step. A one-step experiment uses the base rate so tiny teaching fixtures still learn.
    """

    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    if optimizer_step < 0 or optimizer_step >= total_optimizer_steps:
        raise ValueError("optimizer_step is outside the configured run")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if total_optimizer_steps == 1:
        return base_learning_rate

    warmup_steps = math.ceil(total_optimizer_steps * warmup_ratio) if warmup_ratio else 0
    if warmup_steps and optimizer_step < warmup_steps:
        return base_learning_rate * (optimizer_step + 1) / warmup_steps
    decay_steps = total_optimizer_steps - warmup_steps
    if decay_steps <= 1:
        return base_learning_rate
    progress = (optimizer_step - warmup_steps) / (decay_steps - 1)
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def optimizer_groups(
    example_count: int,
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    seed: int,
    max_steps: int | None,
) -> list[list[tuple[int, ...]]]:
    """Build the complete deterministic batch schedule grouped by optimizer boundary.

    Each outer item is one optimizer step. Its inner tuples are micro-batches. Regenerating this
    schedule and skipping completed outer items is the clean-boundary resume contract.
    """

    if example_count <= 0:
        raise ValueError("example_count must be positive")
    if micro_batch_size <= 0 or gradient_accumulation_steps <= 0 or epochs <= 0:
        raise ValueError("batch size, accumulation steps, and epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")

    micro_batches: list[tuple[int, ...]] = []
    for epoch in range(epochs):
        indices = list(range(example_count))
        random.Random(f"nanopt-sft-v1:{seed}:{epoch}").shuffle(indices)
        micro_batches.extend(
            tuple(indices[start : start + micro_batch_size])
            for start in range(0, example_count, micro_batch_size)
        )
    groups = [
        micro_batches[start : start + gradient_accumulation_steps]
        for start in range(0, len(micro_batches), gradient_accumulation_steps)
    ]
    return groups[:max_steps] if max_steps is not None else groups


def select_examples(examples: Sequence[T], indices: Sequence[int]) -> list[T]:
    """Materialize one scheduled micro-batch with bounds checked by Python indexing."""

    return [examples[index] for index in indices]
