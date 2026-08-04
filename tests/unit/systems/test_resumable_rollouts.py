from __future__ import annotations

from dataclasses import replace

import pytest

from nanopt.systems.resumable_rollouts import (
    simulate_resumable_rollouts,
    validate_resume_checkpoint,
)


def _simulate(sync_mode: str, *, max_policy_lag: int = 0):  # type: ignore[no-untyped-def]
    return simulate_resumable_rollouts(
        [2, 8, 2, 2],
        worker_count=2,
        update_every_completions=1,
        tool_budget=10,
        max_policy_lag=max_policy_lag,
        external_cache_capacity=4,
        sync_mode=sync_mode,  # type: ignore[arg-type]
    )


def test_weight_sync_boundary_exposes_staleness_cache_and_mixed_policy_tradeoff() -> None:
    episode_boundary = _simulate("episode_boundary")
    action_boundary = _simulate("action_boundary")

    assert episode_boundary.ticks == action_boundary.ticks == 8
    assert episode_boundary.policy_updates == action_boundary.policy_updates == 3
    assert len(episode_boundary.checkpoints) == len(action_boundary.checkpoints) == 3

    assert episode_boundary.mixed_policy_trajectories == 0
    assert episode_boundary.stale_trajectories == 1
    assert episode_boundary.cache.hits == 3
    assert episode_boundary.cache.recomputed_prompt_tokens == 0

    assert action_boundary.mixed_policy_trajectories == 1
    assert action_boundary.stale_trajectories == 1
    assert action_boundary.cache.hits == 0
    assert action_boundary.cache.misses == 3
    assert action_boundary.cache.recomputed_prompt_tokens == 42
    assert not episode_boundary.used_for_model_update
    assert not action_boundary.used_for_model_update


def test_action_boundary_records_each_behavior_policy_and_bounded_suffix() -> None:
    simulation = _simulate("action_boundary", max_policy_lag=1)
    long_episode = next(
        item for item in simulation.admissions if item.trajectory_id == "trajectory-1"
    )

    assert long_episode.action_policy_versions == (0, 0, 1, 1, 2, 2, 3, 3)
    assert long_episode.mixed_policy_versions
    assert not long_episode.strict_episode_eligible
    assert long_episode.bounded_action_eligible == 4
    assert long_episode.bounded_action_rejected == 4


def test_checkpoint_binds_model_cursor_world_cursor_snapshot_and_payload() -> None:
    checkpoint = _simulate("episode_boundary").checkpoints[0]
    validate_resume_checkpoint(
        checkpoint,
        expected_snapshot_sha256=checkpoint.world.initial_snapshot_sha256,
    )

    bad_world = replace(checkpoint.world, event_cursor=checkpoint.world.event_cursor + 1)
    with pytest.raises(ValueError, match="cursors disagree"):
        validate_resume_checkpoint(
            replace(checkpoint, world=bad_world),
            expected_snapshot_sha256=checkpoint.world.initial_snapshot_sha256,
        )

    with pytest.raises(ValueError, match="snapshot identity"):
        validate_resume_checkpoint(checkpoint, expected_snapshot_sha256="f" * 64)

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_resume_checkpoint(
            replace(checkpoint, tick=checkpoint.tick + 1),
            expected_snapshot_sha256=checkpoint.world.initial_snapshot_sha256,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lengths", [], "lengths"),
        ("worker_count", 0, "worker_count"),
        ("update_every_completions", 0, "update_every"),
        ("tool_budget", 1, "tool_budget"),
        ("max_policy_lag", -1, "max_policy_lag"),
        ("external_cache_capacity", -1, "external_cache_capacity"),
    ],
)
def test_invalid_simulation_boundaries_are_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "lengths": [2, 8, 2, 2],
        "worker_count": 2,
        "update_every_completions": 1,
        "tool_budget": 10,
        "max_policy_lag": 0,
        "external_cache_capacity": 4,
        "sync_mode": "episode_boundary",
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        simulate_resumable_rollouts(**arguments)  # type: ignore[arg-type]
