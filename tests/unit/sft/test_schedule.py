from __future__ import annotations

import pytest

from nanopt.sft.schedule import cosine_learning_rate, optimizer_groups


def test_optimizer_groups_are_stable_and_stop_only_at_boundaries() -> None:
    first = optimizer_groups(
        5,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        epochs=2,
        seed=42,
        max_steps=None,
    )
    repeated = optimizer_groups(
        5,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        epochs=2,
        seed=42,
        max_steps=None,
    )

    assert first == repeated
    assert len(first) == 3
    assert all(1 <= len(group) <= 2 for group in first)
    assert sorted(index for group in first for batch in group for index in batch) == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]


def test_cosine_schedule_exposes_warmup_and_zero_endpoint() -> None:
    rates = [
        cosine_learning_rate(
            step,
            total_optimizer_steps=5,
            warmup_ratio=0.2,
            base_learning_rate=1.0,
        )
        for step in range(5)
    ]

    assert rates[0] == 1.0
    assert rates[-1] == pytest.approx(0.0)
    assert rates[1] > rates[2] > rates[3] > rates[4]
