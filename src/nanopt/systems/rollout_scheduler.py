"""Small deterministic simulation of long-tail rollouts and policy staleness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RefreshPolicy = Literal["finish_stale", "restart_partial"]


@dataclass(frozen=True)
class RolloutResult:
    """One completed job and the policy versions that bounded its generation."""

    job_id: int
    required_steps: int
    start_policy_version: int
    finish_policy_version: int
    staleness: int
    discarded_steps: int
    finished_tick: int


@dataclass(frozen=True)
class RolloutSimulation:
    """Aggregate queue behavior plus inspectable per-job results."""

    refresh_policy: RefreshPolicy
    worker_count: int
    update_every: int
    ticks: int
    policy_updates: int
    useful_steps: int
    discarded_steps: int
    stale_completions: int
    maximum_staleness: int
    results: tuple[RolloutResult, ...]


@dataclass
class _ActiveRollout:
    job_id: int
    required_steps: int
    completed_steps: int
    start_policy_version: int
    discarded_steps: int = 0


def simulate_rollouts(
    lengths: list[int],
    *,
    worker_count: int,
    update_every: int,
    refresh_policy: RefreshPolicy,
) -> RolloutSimulation:
    """Simulate synchronous policy updates competing with variable rollout lengths.

    Every tick advances each active rollout by one abstract generation step. After
    ``update_every`` completions, the policy version increments. ``finish_stale`` lets active jobs
    complete under their original version; ``restart_partial`` discards their partial work and
    restarts them under the new version. The simulation deliberately omits real token generation,
    networking, and KV-cache implementation so the staleness/throughput tradeoff stays visible.
    """

    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("rollout lengths must be a non-empty list of positive integers")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if update_every <= 0:
        raise ValueError("update_every must be positive")
    if refresh_policy not in {"finish_stale", "restart_partial"}:
        raise ValueError(f"unknown refresh policy: {refresh_policy!r}")

    pending = list(enumerate(lengths))
    active: list[_ActiveRollout] = []
    results: list[RolloutResult] = []
    policy_version = 0
    completions_since_update = 0
    ticks = 0

    def fill_workers() -> None:
        while pending and len(active) < worker_count:
            job_id, required_steps = pending.pop(0)
            active.append(
                _ActiveRollout(
                    job_id=job_id,
                    required_steps=required_steps,
                    completed_steps=0,
                    start_policy_version=policy_version,
                )
            )

    fill_workers()
    while active:
        ticks += 1
        for job in active:
            job.completed_steps += 1

        finished = [job for job in active if job.completed_steps == job.required_steps]
        for job in finished:
            staleness = policy_version - job.start_policy_version
            results.append(
                RolloutResult(
                    job_id=job.job_id,
                    required_steps=job.required_steps,
                    start_policy_version=job.start_policy_version,
                    finish_policy_version=policy_version,
                    staleness=staleness,
                    discarded_steps=job.discarded_steps,
                    finished_tick=ticks,
                )
            )
            active.remove(job)
        completions_since_update += len(finished)

        work_remains = bool(active or pending)
        if completions_since_update >= update_every and work_remains:
            policy_version += 1
            completions_since_update = 0
            if refresh_policy == "restart_partial":
                for job in active:
                    job.discarded_steps += job.completed_steps
                    job.completed_steps = 0
                    job.start_policy_version = policy_version
        fill_workers()

    ordered = tuple(sorted(results, key=lambda result: result.job_id))
    return RolloutSimulation(
        refresh_policy=refresh_policy,
        worker_count=worker_count,
        update_every=update_every,
        ticks=ticks,
        policy_updates=policy_version,
        useful_steps=sum(lengths),
        discarded_steps=sum(result.discarded_steps for result in ordered),
        stale_completions=sum(result.staleness > 0 for result in ordered),
        maximum_staleness=max(result.staleness for result in ordered),
        results=ordered,
    )
