from __future__ import annotations

import pytest

from nanopt.systems.rollout_scheduler import simulate_rollouts


def test_finish_stale_and_restart_partial_show_the_tradeoff() -> None:
    lengths = [2, 8, 2, 2]
    stale = simulate_rollouts(
        lengths,
        worker_count=2,
        update_every=1,
        refresh_policy="finish_stale",
    )
    restarted = simulate_rollouts(
        lengths,
        worker_count=2,
        update_every=1,
        refresh_policy="restart_partial",
    )

    assert stale.useful_steps == restarted.useful_steps == sum(lengths)
    assert stale.stale_completions > 0
    assert stale.discarded_steps == 0
    assert restarted.stale_completions == 0
    assert restarted.discarded_steps > 0
    assert restarted.ticks > stale.ticks


@pytest.mark.parametrize("lengths", [[], [1, 0], [-1]])
def test_invalid_lengths_are_rejected(lengths: list[int]) -> None:
    with pytest.raises(ValueError, match="lengths"):
        simulate_rollouts(
            lengths,
            worker_count=1,
            update_every=1,
            refresh_policy="finish_stale",
        )
