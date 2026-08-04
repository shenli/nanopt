"""Artifact-producing entry point for the v0.4 resumable-rollout systems lab."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from nanopt.config.models import SystemsLabExperiment
from nanopt.config.resolver import ResolutionResult
from nanopt.runtime.artifacts import append_jsonl, sha256_file, write_json, write_text
from nanopt.runtime.run_context import RunContext, create_run_context
from nanopt.systems.resumable_rollouts import (
    ResumableRolloutSimulation,
    simulate_resumable_rollouts,
)


def _headline(simulation: ResumableRolloutSimulation) -> dict[str, int | str | bool]:
    return {
        "sync_mode": simulation.sync_mode,
        "ticks": simulation.ticks,
        "policy_updates": simulation.policy_updates,
        "completed_trajectories": simulation.completed_trajectories,
        "partial_checkpoints": len(simulation.checkpoints),
        "strict_eligible_trajectories": simulation.strict_eligible_trajectories,
        "mixed_policy_trajectories": simulation.mixed_policy_trajectories,
        "stale_trajectories": simulation.stale_trajectories,
        "external_cache_hits": simulation.cache.hits,
        "external_cache_misses": simulation.cache.misses,
        "recomputed_prompt_tokens": simulation.cache.recomputed_prompt_tokens,
        "used_for_model_update": simulation.used_for_model_update,
    }


def _report(
    experiment: SystemsLabExperiment,
    simulations: list[ResumableRolloutSimulation],
) -> str:
    rows = "\n".join(
        "| "
        f"`{item.sync_mode}` | {item.ticks} | {len(item.checkpoints)} | "
        f"{item.strict_eligible_trajectories}/{item.completed_trajectories} | "
        f"{item.mixed_policy_trajectories} | {item.stale_trajectories} | "
        f"{item.cache.hits}/{item.cache.misses} | "
        f"{item.cache.recomputed_prompt_tokens} |"
        for item in simulations
    )
    return f"""# Resumable rollout systems simulation

## Scope

This CPU run used the deterministic v0.4 control-plane simulation. It did not load Qwen, allocate
a KV cache, run a sandbox, measure throughput, or update a model. Synthetic exact token IDs and
workspace hashes make pause/resume and policy identity inspectable without claiming production
performance.

## Configuration

- Rollout action lengths: `{experiment.workload.rollout_action_lengths}`
- Workers: `{experiment.workload.worker_count}`
- Publish after completions: `{experiment.workload.update_every_completions}`
- Strict maximum policy lag: `{experiment.freshness.max_policy_lag}`
- External cache capacity: `{experiment.cache.capacity_entries}` entries

## Comparison

| Sync | Ticks | Checkpoints | Eligible | Mixed | Stale | Cache hit/miss | Recomputed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

`episode_boundary` preserves a single behavior policy for a long episode and can reuse prefix
state under the same weights, but the episode becomes stale as newer policies are published.
`action_boundary` refreshes only between complete tool actions, which avoids splitting an action
across weights; it creates a mixed-policy episode and invalidates old-policy KV entries. Neither
case is silently admitted to NanoPT's strict fresh-episode objective.

## Artifact guide

- `actions.jsonl` gives exact synthetic prompt/action coordinates and behavior-policy identity.
- `partial_checkpoints.jsonl` pairs hash-bound model execution state with world state.
- `weight_sync_events.jsonl` records policy publication and cache-reuse decisions.
- `admission_decisions.jsonl` explains fresh, stale, and mixed-policy rejection.
- `summary.json` contains the compact comparison used above.
"""


def _record_artifact(context: RunContext, filename: str, kind: str) -> None:
    path = context.run_dir / filename
    context.manifest["artifacts"].append(
        {"path": filename, "kind": kind, "sha256": sha256_file(path)}
    )


def execute_systems_lab_run(
    result: ResolutionResult,
    *,
    artifacts_root: Path = Path("artifacts/runs"),
    run_id: str | None = None,
) -> RunContext:
    """Run both safe-boundary strategies and retain their inspectable control-plane evidence."""

    experiment = result.config.experiment
    if not isinstance(experiment, SystemsLabExperiment):
        raise TypeError("systems run requires a systems_lab experiment profile")

    context = create_run_context(result, artifacts_root=artifacts_root, run_id=run_id)
    context.set_status("running")
    try:
        simulations = [
            simulate_resumable_rollouts(
                experiment.workload.rollout_action_lengths,
                worker_count=experiment.workload.worker_count,
                update_every_completions=experiment.workload.update_every_completions,
                tool_budget=experiment.workload.tool_budget,
                max_policy_lag=experiment.freshness.max_policy_lag,
                external_cache_capacity=experiment.cache.capacity_entries,
                sync_mode=mode,
            )
            for mode in experiment.weight_sync.compare_modes
        ]

        for simulation in simulations:
            for action in simulation.actions:
                append_jsonl(
                    context.run_dir / "actions.jsonl",
                    {"sync_mode": simulation.sync_mode, **asdict(action)},
                )
            for checkpoint in simulation.checkpoints:
                append_jsonl(
                    context.run_dir / "partial_checkpoints.jsonl",
                    {"sync_mode": simulation.sync_mode, **asdict(checkpoint)},
                )
            for event in simulation.weight_sync_events:
                append_jsonl(
                    context.run_dir / "weight_sync_events.jsonl",
                    {"sync_mode": simulation.sync_mode, **asdict(event)},
                )
            for decision in simulation.admissions:
                append_jsonl(
                    context.run_dir / "admission_decisions.jsonl",
                    {"sync_mode": simulation.sync_mode, **asdict(decision)},
                )

        summary: dict[str, Any] = {
            "schema_version": 1,
            "status": "v0_4_systems_simulation_passed",
            "backend": experiment.backend.rollout_backend,
            "measured_throughput_claim": experiment.backend.measured_throughput_claim,
            "simulated_experience_used_for_update": (
                experiment.freshness.simulated_experience_used_for_update
            ),
            "comparisons": [_headline(item) for item in simulations],
        }
        write_json(context.run_dir / "summary.json", summary)
        write_text(context.run_dir / "report.md", _report(experiment, simulations))

        for filename, kind in (
            ("actions.jsonl", "systems_synthetic_actions"),
            ("partial_checkpoints.jsonl", "systems_partial_checkpoints"),
            ("weight_sync_events.jsonl", "systems_weight_sync_events"),
            ("admission_decisions.jsonl", "systems_admission_decisions"),
            ("summary.json", "systems_summary"),
            ("report.md", "systems_report"),
        ):
            _record_artifact(context, filename, kind)
        context.set_status("completed")
    except Exception as exc:
        context.set_status(
            "failed",
            failure={"type": type(exc).__name__, "message": str(exc), "phase": "simulation"},
        )
        raise
    return context
