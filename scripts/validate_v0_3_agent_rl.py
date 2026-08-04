"""Validate grouped exact-token Agent RL, studies, Docker isolation, and GPU evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema

from nanopt.agent.rl_records import (
    AgentRlBudgetStudy,
    AgentRlCreditStudy,
    AgentRlGroup,
    AgentRlStalenessStudy,
    AgentRlSummary,
)
from nanopt.config.loader import ConfigRepository
from nanopt.runtime.artifacts import read_jsonl, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata
from nanopt.sft.checkpoint import sha256_directory
from scripts.validate_m3_reference_smoke import _read_object, _require, _validate_doctor
from scripts.validate_m8_reference_agent import _verify_checksums


def validate_v0_3_agent_rl(evidence_root: Path, project_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    project_root = project_root.resolve()
    doctor = _validate_doctor(evidence_root, project_root)
    _require(doctor["docker"]["daemon_reachable"] is True, "Docker daemon was unreachable")
    checksums = _verify_checksums(evidence_root)

    run_dir = evidence_root / "runs/agent-rl"
    manifest = _read_object(run_dir / "run_manifest.json")
    run_schema = _read_object(project_root / "specs/schemas/run_manifest.schema.json")
    jsonschema.Draft202012Validator(run_schema).validate(manifest)
    _require(manifest["stage"] == "agent_rl", "training stage differs")
    _require(manifest["status"] == "completed", "Agent RL run did not complete")
    _require(manifest["git"]["dirty"] is False, "Agent RL used a dirty checkout")
    _require(
        manifest["git"]["commit"] == collect_git_metadata(project_root)["commit"],
        "Agent RL commit differs",
    )
    _require(manifest["agent_environment"]["backend"] == "docker", "Agent RL was not Docker")
    _require(
        manifest["agent_environment"]["hidden_source_exposed"] is False,
        "hidden source was marked exposed",
    )
    _require(
        manifest["training"]["consumed_exact_stored_token_ids"] is True,
        "Agent RL did not declare exact stored token consumption",
    )
    _require(manifest["training"]["maximum_policy_lag"] == 0, "training policy lag differs")
    _require(
        manifest["training"]["hidden_reward_exposed_during_rollout"] is False,
        "hidden reward was marked exposed during rollout",
    )

    summary = AgentRlSummary.model_validate(_read_object(run_dir / "summary.json"), strict=True)
    summary_schema = _read_object(project_root / "specs/schemas/agent_rl_summary.schema.json")
    jsonschema.Draft202012Validator(summary_schema).validate(summary.model_dump(mode="json"))
    groups = [
        AgentRlGroup.model_validate(value, strict=True)
        for value in read_jsonl(run_dir / "rollout_groups.jsonl")
    ]
    group_schema = _read_object(project_root / "specs/schemas/agent_rl_group.schema.json")
    action_schema = _read_object(project_root / "specs/schemas/agent_rl_action.schema.json")
    for group in groups:
        jsonschema.Draft202012Validator(group_schema).validate(group.model_dump(mode="json"))
        for episode in group.episodes:
            for action in episode.actions:
                jsonschema.Draft202012Validator(action_schema).validate(
                    action.model_dump(mode="json")
                )
                _require(action.reference_logprobs is not None, "reference logps are missing")
                _require(
                    len(action.sampled_token_ids)
                    == len(action.action_mask)
                    == len(action.old_logprobs)
                    == len(action.reference_logprobs),
                    "Agent RL action coordinates differ",
                )
    _require(len(groups) == summary.groups == summary.iterations, "group count differs")
    _require(
        sum(len(group.episodes) for group in groups) == summary.episodes,
        "episode count differs",
    )
    _require(
        sum(len(episode.actions) for group in groups for episode in group.episodes)
        == summary.actions,
        "action count differs",
    )
    _require(summary.maximum_training_policy_lag == 0, "summary policy lag differs")
    _require(summary.hidden_reward_exposed_during_rollout is False, "summary reward leak differs")
    _require(summary.degenerate_group_fraction < 1.0, "all Agent RL groups were degenerate")

    staleness = AgentRlStalenessStudy.model_validate(
        _read_object(run_dir / "staleness_study.json"), strict=True
    )
    _require(staleness.fresh.used_for_update is False, "fresh study point entered training")
    _require(staleness.stale.used_for_update is False, "stale study point entered training")
    _require(staleness.fresh.policy_lag == 1, "fresh study lag differs")
    _require(
        staleness.stale.policy_lag == summary.iterations,
        "stale study lag does not span the run",
    )

    credit = AgentRlCreditStudy.model_validate(
        _read_object(run_dir / "credit_study.json"), strict=True
    )
    _require(
        credit.all_actions_active_tokens > credit.terminal_action_active_tokens,
        "credit-assignment coverage did not differ",
    )
    _require(
        sum(credit.active_tokens_by_tool.values()) == credit.all_actions_active_tokens,
        "tool credit counts do not cover all action tokens",
    )

    budget = AgentRlBudgetStudy.model_validate(
        _read_object(run_dir / "tool_budget_study.json"), strict=True
    )
    checkpoints = {point.checkpoint for point in budget.points}
    budgets = {point.tool_budget for point in budget.points}
    _require(checkpoints == {"reference", "agent_rl"}, "budget study checkpoints differ")
    _require(len(budgets) >= 2, "budget study needs at least two caps")
    maximum_budget = max(budgets)
    reference_full = next(
        point
        for point in budget.points
        if point.checkpoint == "reference" and point.tool_budget == maximum_budget
    )
    policy_full = next(
        point
        for point in budget.points
        if point.checkpoint == "agent_rl" and point.tool_budget == maximum_budget
    )
    _require(
        policy_full.mean_hidden_outcome_reward >= reference_full.mean_hidden_outcome_reward,
        "Agent RL regressed the held-out full-budget reward",
    )

    hardware = ConfigRepository(project_root / "configs").hardware("rtx_4070_ti_super_16gb")
    hard_bytes = int(hardware.memory_budget.hard_peak_reserved_gib * 1024**3)
    _require(summary.peak_reserved_bytes <= hard_bytes, "Agent RL exceeded hard VRAM")
    adapter_dir = run_dir / "adapter" / "agent_rl"
    _require(adapter_dir.is_dir(), "Agent RL adapter directory is missing")
    _require(
        sha256_directory(adapter_dir) == summary.agent_rl_adapter_sha256, "adapter hash differs"
    )

    return {
        "schema_version": 1,
        "status": "v0_3_agent_rl_passed",
        "git_commit": collect_git_metadata(project_root)["commit"],
        "hardware": {
            "profile": hardware.id,
            "gpu": doctor["cuda"]["gpus"][0]["name"],
            "peak_reserved_gib": summary.peak_reserved_bytes / 1024**3,
            "hard_limit_gib": hardware.memory_budget.hard_peak_reserved_gib,
        },
        "training": {
            "iterations": summary.iterations,
            "optimizer_steps": summary.optimizer_steps,
            "groups": summary.groups,
            "episodes": summary.episodes,
            "actions": summary.actions,
            "mean_reward": summary.mean_reward,
            "action_validity_rate": summary.action_validity_rate,
            "degenerate_group_fraction": summary.degenerate_group_fraction,
            "initial_validation_reward": summary.initial_validation_reward,
            "final_validation_reward": summary.final_validation_reward,
            "parent_adapter_sha256": summary.parent_agent_sft_adapter_sha256,
            "adapter_sha256": summary.agent_rl_adapter_sha256,
        },
        "staleness": staleness.model_dump(mode="json"),
        "credit_assignment": credit.model_dump(mode="json"),
        "tool_budget": budget.model_dump(mode="json"),
        "artifacts": {
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "checksums_file_count": len(checksums),
            "checksums_manifest_sha256": sha256_file(evidence_root / "checksums.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = validate_v0_3_agent_rl(args.evidence_root, args.project_root)
    if args.output:
        write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
