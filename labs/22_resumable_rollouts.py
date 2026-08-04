"""Trace pause/resume, weight publication, and cache identity without a cluster."""

from __future__ import annotations

from nanopt.systems.resumable_rollouts import simulate_resumable_rollouts


def main() -> None:
    """Compare the two safe points where a rollout worker may synchronize weights."""

    common = {
        "worker_count": 2,
        "update_every_completions": 1,
        "tool_budget": 10,
        "max_policy_lag": 0,
        "external_cache_capacity": 4,
    }
    episode_boundary = simulate_resumable_rollouts(
        [2, 8, 2, 2],
        sync_mode="episode_boundary",
        **common,
    )
    action_boundary = simulate_resumable_rollouts(
        [2, 8, 2, 2],
        sync_mode="action_boundary",
        **common,
    )

    print("mode              mixed  stale  cache hit/miss  recomputed prompt tokens")
    for result in (episode_boundary, action_boundary):
        print(
            f"{result.sync_mode:17} "
            f"{result.mixed_policy_trajectories:>5} "
            f"{result.stale_trajectories:>6} "
            f"{result.cache.hits:>5}/{result.cache.misses:<4} "
            f"{result.cache.recomputed_prompt_tokens:>24}"
        )

    assert episode_boundary.cache.hits == 3
    assert episode_boundary.stale_trajectories == 1
    assert action_boundary.mixed_policy_trajectories == 1
    assert action_boundary.cache.recomputed_prompt_tokens == 42
    assert all(not result.used_for_model_update for result in (episode_boundary, action_boundary))
    print("Resumable-rollout systems lab passed.")


if __name__ == "__main__":
    main()
