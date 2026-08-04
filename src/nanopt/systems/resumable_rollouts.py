"""Deterministic partial-rollout, checkpoint, cache, and policy-age simulation.

The v0.4 systems laboratory deliberately simulates control-plane decisions instead of pretending
to be a rollout server.  Token IDs, cache entries, and workspace hashes are synthetic but exact;
the scheduler invariants are the same ones a real runtime must make explicit.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from typing import Literal

from nanopt.runtime.artifacts import canonical_json, sha256_bytes

WeightSyncMode = Literal["episode_boundary", "action_boundary"]


def _hash_record(value: object) -> str:
    """Hash one JSON-compatible record with NanoPT's canonical serialization."""

    return sha256_bytes(canonical_json(value))


def _policy_hash(version: int) -> str:
    return sha256_bytes(f"synthetic-policy-version:{version}".encode())


def _workspace_hash(snapshot_sha256: str, event_cursor: int) -> str:
    return _hash_record({"snapshot_sha256": snapshot_sha256, "event_cursor": event_cursor})


def _cache_key(policy_sha256: str, prompt_token_ids: tuple[int, ...]) -> str:
    # KV values depend on both the prefix and the weights that produced them. Reusing an old
    # policy's cache after a weight sync would be numerically inconsistent.
    return _hash_record(
        {"policy_sha256": policy_sha256, "prompt_token_ids": list(prompt_token_ids)}
    )


@dataclass(frozen=True)
class SyntheticActionSegment:
    """One complete simulated tool action with exact model and world coordinates.

    ``prompt_token_ids`` and ``sampled_token_ids`` are deterministic teaching IDs, not tokens from
    Qwen.  The record exists to show where exact IDs and behavior-policy identity cross a systems
    boundary.  Real v0.3 Agent RL records additionally retain FP32 behavior/reference logprobs.
    """

    trajectory_id: str
    action_index: int
    behavior_policy_version: int
    behavior_policy_sha256: str
    prompt_token_ids: tuple[int, ...]
    sampled_token_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    workspace_before_sha256: str
    workspace_after_sha256: str


@dataclass(frozen=True)
class ModelExecutionState:
    """Model-side state required to resume between complete tool actions."""

    prompt_token_ids: tuple[int, ...]
    sampling_rng_counter: int
    stop_state: Literal["between_actions"]
    behavior_policy_version: int
    behavior_policy_sha256: str
    generation_config_sha256: str
    prefix_cache_key: str


@dataclass(frozen=True)
class WorldExecutionState:
    """Environment-side state paired with one model execution checkpoint."""

    initial_snapshot_sha256: str
    workspace_sha256: str
    event_cursor: int
    remaining_tool_budget: int


@dataclass(frozen=True)
class PartialRolloutCheckpoint:
    """Hash-bound model and world state captured at an action boundary."""

    schema_version: int
    checkpoint_id: str
    trajectory_id: str
    tick: int
    model: ModelExecutionState
    world: WorldExecutionState
    payload_sha256: str


def _checkpoint_payload(checkpoint: PartialRolloutCheckpoint) -> dict[str, object]:
    value = asdict(checkpoint)
    value.pop("payload_sha256")
    return value


def validate_resume_checkpoint(
    checkpoint: PartialRolloutCheckpoint,
    *,
    expected_snapshot_sha256: str,
) -> None:
    """Reject a checkpoint whose model/world boundary or hash no longer agrees.

    Resume is allowed only between complete actions. The model RNG counter and world event cursor
    must point at the same next action, and the workspace must derive from the expected immutable
    task snapshot. These checks prevent a plausible transcript from being paired with the wrong
    external state.
    """

    if checkpoint.schema_version != 1:
        raise ValueError("unsupported partial-rollout checkpoint schema")
    if checkpoint.model.stop_state != "between_actions":
        raise ValueError("partial rollout may resume only between complete actions")
    if checkpoint.world.initial_snapshot_sha256 != expected_snapshot_sha256:
        raise ValueError("partial-rollout snapshot identity changed before resume")
    if checkpoint.model.sampling_rng_counter != checkpoint.world.event_cursor:
        raise ValueError("model RNG and world event cursors disagree")
    expected_workspace = _workspace_hash(
        checkpoint.world.initial_snapshot_sha256,
        checkpoint.world.event_cursor,
    )
    if checkpoint.world.workspace_sha256 != expected_workspace:
        raise ValueError("checkpoint workspace hash does not match its event cursor")
    if checkpoint.world.remaining_tool_budget < 0:
        raise ValueError("checkpoint exhausted its tool budget before resume")
    if checkpoint.payload_sha256 != _hash_record(_checkpoint_payload(checkpoint)):
        raise ValueError("partial-rollout checkpoint hash mismatch")


@dataclass(frozen=True)
class WeightSyncEvent:
    """One worker decision after the trainer publishes a policy version."""

    trajectory_id: str
    published_policy_version: int
    previous_worker_policy_version: int
    resumed_worker_policy_version: int
    sync_mode: WeightSyncMode
    cache_reusable: bool


@dataclass(frozen=True)
class AdmissionDecision:
    """Explain whether one completed episode is safe for a strict fresh update."""

    trajectory_id: str
    trainer_policy_version: int
    action_policy_versions: tuple[int, ...]
    mixed_policy_versions: bool
    maximum_policy_lag: int
    strict_episode_eligible: bool
    bounded_action_eligible: int
    bounded_action_rejected: int
    reason: str


@dataclass(frozen=True)
class CacheMetrics:
    """External-prefix-cache behavior observed during one simulation."""

    writes: int
    hits: int
    misses: int
    evictions: int
    recomputed_prompt_tokens: int


@dataclass(frozen=True)
class ResumableRolloutSimulation:
    """Complete evidence for one weight-synchronization strategy."""

    schema_version: int
    sync_mode: WeightSyncMode
    backend: Literal["deterministic_simulation"]
    ticks: int
    policy_updates: int
    completed_trajectories: int
    checkpoints: tuple[PartialRolloutCheckpoint, ...]
    actions: tuple[SyntheticActionSegment, ...]
    weight_sync_events: tuple[WeightSyncEvent, ...]
    admissions: tuple[AdmissionDecision, ...]
    cache: CacheMetrics
    strict_eligible_trajectories: int
    mixed_policy_trajectories: int
    stale_trajectories: int
    used_for_model_update: Literal[False]


@dataclass
class _ActiveTrajectory:
    trajectory_id: str
    job_id: int
    required_actions: int
    completed_actions: int
    worker_policy_version: int
    snapshot_sha256: str
    prefix_token_ids: list[int]
    actions: list[SyntheticActionSegment]


class _ExternalPrefixCache:
    """Tiny LRU metadata cache representing CPU-resident prefix state."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: OrderedDict[str, int] = OrderedDict()
        self.writes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def write(self, key: str, token_count: int) -> None:
        if self.capacity == 0:
            return
        self.writes += 1
        self.entries.pop(key, None)
        self.entries[key] = token_count
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
            self.evictions += 1

    def restore(self, key: str) -> bool:
        token_count = self.entries.pop(key, None)
        if token_count is None:
            self.misses += 1
            return False
        self.entries[key] = token_count
        self.hits += 1
        return True


def _make_checkpoint(
    job: _ActiveTrajectory,
    *,
    tick: int,
    tool_budget: int,
    generation_config_sha256: str,
) -> PartialRolloutCheckpoint:
    policy_sha256 = _policy_hash(job.worker_policy_version)
    model = ModelExecutionState(
        prompt_token_ids=tuple(job.prefix_token_ids),
        sampling_rng_counter=job.completed_actions,
        stop_state="between_actions",
        behavior_policy_version=job.worker_policy_version,
        behavior_policy_sha256=policy_sha256,
        generation_config_sha256=generation_config_sha256,
        prefix_cache_key=_cache_key(policy_sha256, tuple(job.prefix_token_ids)),
    )
    world = WorldExecutionState(
        initial_snapshot_sha256=job.snapshot_sha256,
        workspace_sha256=_workspace_hash(job.snapshot_sha256, job.completed_actions),
        event_cursor=job.completed_actions,
        remaining_tool_budget=tool_budget - job.completed_actions,
    )
    empty_hash = "0" * 64
    checkpoint = PartialRolloutCheckpoint(
        schema_version=1,
        checkpoint_id=f"{job.trajectory_id}-tick-{tick}",
        trajectory_id=job.trajectory_id,
        tick=tick,
        model=model,
        world=world,
        payload_sha256=empty_hash,
    )
    return replace(checkpoint, payload_sha256=_hash_record(_checkpoint_payload(checkpoint)))


def _append_action(job: _ActiveTrajectory) -> None:
    action_index = job.completed_actions
    prompt = tuple(job.prefix_token_ids)
    sampled = (10_000 + job.job_id * 100 + action_index, 20_000 + action_index)
    before = _workspace_hash(job.snapshot_sha256, action_index)
    after = _workspace_hash(job.snapshot_sha256, action_index + 1)
    version = job.worker_policy_version
    job.actions.append(
        SyntheticActionSegment(
            trajectory_id=job.trajectory_id,
            action_index=action_index,
            behavior_policy_version=version,
            behavior_policy_sha256=_policy_hash(version),
            prompt_token_ids=prompt,
            sampled_token_ids=sampled,
            action_mask=(1, 1),
            workspace_before_sha256=before,
            workspace_after_sha256=after,
        )
    )
    # A deterministic synthetic tool observation becomes part of the next prompt. Keeping it
    # explicit makes prefix growth and cache-recomputation cost visible to the learner.
    job.prefix_token_ids.extend((*sampled, 30_000 + action_index))
    job.completed_actions += 1


def _admit_episode(
    job: _ActiveTrajectory,
    *,
    trainer_policy_version: int,
    max_policy_lag: int,
) -> AdmissionDecision:
    versions = tuple(action.behavior_policy_version for action in job.actions)
    unique_versions = set(versions)
    lags = tuple(trainer_policy_version - version for version in versions)
    if any(lag < 0 for lag in lags):
        raise ValueError("action policy version is newer than the trainer")
    maximum_lag = max(lags)
    mixed = len(unique_versions) > 1
    strict = not mixed and maximum_lag <= max_policy_lag
    bounded_eligible = sum(lag <= max_policy_lag for lag in lags)
    if mixed:
        reason = "mixed behavior-policy versions require a segmented/off-policy objective"
    elif maximum_lag > max_policy_lag:
        reason = f"episode policy lag {maximum_lag} exceeds limit {max_policy_lag}"
    else:
        reason = "one behavior policy and bounded policy lag"
    return AdmissionDecision(
        trajectory_id=job.trajectory_id,
        trainer_policy_version=trainer_policy_version,
        action_policy_versions=versions,
        mixed_policy_versions=mixed,
        maximum_policy_lag=maximum_lag,
        strict_episode_eligible=strict,
        bounded_action_eligible=bounded_eligible,
        bounded_action_rejected=len(lags) - bounded_eligible,
        reason=reason,
    )


def simulate_resumable_rollouts(
    lengths: list[int],
    *,
    worker_count: int,
    update_every_completions: int,
    tool_budget: int,
    max_policy_lag: int,
    external_cache_capacity: int,
    sync_mode: WeightSyncMode,
) -> ResumableRolloutSimulation:
    """Simulate partial rollout pause/resume across policy publications.

    Each tick completes one whole synthetic tool action per active worker. An update boundary is
    triggered after ``update_every_completions`` episodes finish. Unfinished episodes are paused
    with hash-bound model/world state, their prefix metadata is offered to a bounded external
    cache, and workers either keep the episode's original weights or synchronize before the next
    action. Completed episodes are classified but never used for a model update.
    """

    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("rollout lengths must be a non-empty list of positive integers")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if update_every_completions <= 0:
        raise ValueError("update_every_completions must be positive")
    if tool_budget <= 0 or any(length > tool_budget for length in lengths):
        raise ValueError("tool_budget must cover every simulated rollout")
    if max_policy_lag < 0:
        raise ValueError("max_policy_lag cannot be negative")
    if external_cache_capacity < 0:
        raise ValueError("external_cache_capacity cannot be negative")
    if sync_mode not in {"episode_boundary", "action_boundary"}:
        raise ValueError(f"unknown weight synchronization mode: {sync_mode!r}")

    generation_config_sha256 = _hash_record({"do_sample": True, "temperature": 1.0, "top_p": 1.0})
    pending = list(enumerate(lengths))
    active: list[_ActiveTrajectory] = []
    checkpoints: list[PartialRolloutCheckpoint] = []
    completed: list[_ActiveTrajectory] = []
    admissions: list[AdmissionDecision] = []
    sync_events: list[WeightSyncEvent] = []
    cache = _ExternalPrefixCache(external_cache_capacity)
    recomputed_prompt_tokens = 0
    policy_version = 0
    completions_since_update = 0
    ticks = 0

    def fill_workers() -> None:
        while pending and len(active) < worker_count:
            job_id, required_actions = pending.pop(0)
            trajectory_id = f"trajectory-{job_id}"
            snapshot = _hash_record({"task": trajectory_id, "initial_state": "clean"})
            active.append(
                _ActiveTrajectory(
                    trajectory_id=trajectory_id,
                    job_id=job_id,
                    required_actions=required_actions,
                    completed_actions=0,
                    worker_policy_version=policy_version,
                    snapshot_sha256=snapshot,
                    prefix_token_ids=[1, 100 + job_id],
                    actions=[],
                )
            )

    fill_workers()
    while active:
        ticks += 1
        for job in active:
            _append_action(job)

        finished = [job for job in active if job.completed_actions == job.required_actions]
        for job in finished:
            admissions.append(
                _admit_episode(
                    job,
                    trainer_policy_version=policy_version,
                    max_policy_lag=max_policy_lag,
                )
            )
            active.remove(job)
            completed.append(job)
        completions_since_update += len(finished)

        work_remains = bool(active or pending)
        if completions_since_update >= update_every_completions and work_remains:
            paused: list[tuple[_ActiveTrajectory, PartialRolloutCheckpoint]] = []
            for job in active:
                checkpoint = _make_checkpoint(
                    job,
                    tick=ticks,
                    tool_budget=tool_budget,
                    generation_config_sha256=generation_config_sha256,
                )
                validate_resume_checkpoint(
                    checkpoint,
                    expected_snapshot_sha256=job.snapshot_sha256,
                )
                checkpoints.append(checkpoint)
                paused.append((job, checkpoint))
                cache.write(
                    checkpoint.model.prefix_cache_key,
                    len(checkpoint.model.prompt_token_ids),
                )

            policy_version += 1
            completions_since_update = 0
            for job, checkpoint in paused:
                previous = job.worker_policy_version
                resumed = policy_version if sync_mode == "action_boundary" else previous
                resumed_policy_sha256 = _policy_hash(resumed)
                resumed_key = _cache_key(
                    resumed_policy_sha256,
                    checkpoint.model.prompt_token_ids,
                )
                cache_hit = cache.restore(resumed_key)
                if not cache_hit:
                    recomputed_prompt_tokens += len(checkpoint.model.prompt_token_ids)
                job.worker_policy_version = resumed
                sync_events.append(
                    WeightSyncEvent(
                        trajectory_id=job.trajectory_id,
                        published_policy_version=policy_version,
                        previous_worker_policy_version=previous,
                        resumed_worker_policy_version=resumed,
                        sync_mode=sync_mode,
                        cache_reusable=cache_hit,
                    )
                )
            fill_workers()

    ordered_completed = sorted(completed, key=lambda job: job.job_id)
    ordered_actions = tuple(action for job in ordered_completed for action in job.actions)
    return ResumableRolloutSimulation(
        schema_version=1,
        sync_mode=sync_mode,
        backend="deterministic_simulation",
        ticks=ticks,
        policy_updates=policy_version,
        completed_trajectories=len(ordered_completed),
        checkpoints=tuple(checkpoints),
        actions=ordered_actions,
        weight_sync_events=tuple(sync_events),
        admissions=tuple(sorted(admissions, key=lambda item: item.trajectory_id)),
        cache=CacheMetrics(
            writes=cache.writes,
            hits=cache.hits,
            misses=cache.misses,
            evictions=cache.evictions,
            recomputed_prompt_tokens=recomputed_prompt_tokens,
        ),
        strict_eligible_trajectories=sum(item.strict_episode_eligible for item in admissions),
        mixed_policy_trajectories=sum(item.mixed_policy_versions for item in admissions),
        stale_trajectories=sum(item.maximum_policy_lag > max_policy_lag for item in admissions),
        used_for_model_update=False,
    )
