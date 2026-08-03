"""Compare stale completion and partial-restart policies without a cluster."""

from __future__ import annotations

from nanopt.systems.rollout_scheduler import simulate_rollouts


def main() -> None:
    """Make the throughput/freshness tradeoff visible with one long-tail queue."""

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

    print("Policy           ticks stale discarded")
    print(
        f"finish_stale     {stale.ticks:>5} {stale.stale_completions:>5} {stale.discarded_steps:>9}"
    )
    print(
        f"restart_partial  {restarted.ticks:>5} {restarted.stale_completions:>5} "
        f"{restarted.discarded_steps:>9}"
    )
    assert stale.stale_completions > 0 and stale.discarded_steps == 0
    assert restarted.stale_completions == 0 and restarted.discarded_steps > 0
    print("Rollout-scheduler simulation passed.")


if __name__ == "__main__":
    main()
