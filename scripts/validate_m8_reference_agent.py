"""Validate M8 task, oracle, baseline, replay, isolation, and Docker security evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema

from nanopt.agent.records import AgentRunSummary, AgentTrajectory
from nanopt.agent.tasks import load_task_suite
from nanopt.config.loader import ConfigRepository
from nanopt.runtime.artifacts import read_jsonl, sha256_file, write_json
from nanopt.runtime.environment import collect_git_metadata
from scripts.validate_m3_reference_smoke import _read_object, _require, _validate_doctor


def _verify_checksums(evidence_root: Path) -> dict[str, str]:
    checksums = _read_object(evidence_root / "checksums.json")
    _require(bool(checksums), "checksum manifest is empty")
    for relative, expected in checksums.items():
        path = evidence_root / str(relative)
        _require(path.is_file(), f"checksummed file is missing: {relative}")
        _require(sha256_file(path) == expected, f"checksum mismatch: {relative}")
    return {str(key): str(value) for key, value in checksums.items()}


def _validate_run_manifest(
    run_dir: Path, project_root: Path, expected_policy: str
) -> dict[str, Any]:
    manifest = _read_object(run_dir / "run_manifest.json")
    schema = _read_object(project_root / "specs/schemas/run_manifest.schema.json")
    jsonschema.Draft202012Validator(schema).validate(manifest)
    _require(manifest["status"] == "completed", f"agent run failed: {run_dir.name}")
    _require(manifest["stage"] == "agent_evaluation", "agent run stage differs")
    _require(manifest["git"]["dirty"] is False, "agent run used a dirty checkout")
    _require(
        manifest["git"]["commit"] == collect_git_metadata(project_root)["commit"],
        "agent run commit differs",
    )
    environment = manifest["agent_environment"]
    _require(environment["backend"] == "docker", "reference agent backend was not Docker")
    _require(environment["policy"] == expected_policy, "agent policy differs")
    _require(environment["network"] == "none", "sandbox network was not disabled")
    _require(environment["run_as_non_root"] is True, "sandbox was not non-root")
    _require(environment["expose_gpu"] is False, "sandbox exposed a GPU")
    _require(environment["capabilities_dropped"] is True, "capabilities were not dropped")
    _require(environment["no_new_privileges"] is True, "no-new-privileges was absent")
    _require(environment["root_filesystem_read_only"] is True, "root filesystem was writable")
    _require(environment["separate_hidden_workspace"] is True, "hidden workspace was shared")
    _require(environment["hidden_source_exposed"] is False, "hidden source was exposed")
    _require(environment["environment_trains_model"] is False, "agent run claimed training")
    return manifest


def _trajectories(run_dir: Path, project_root: Path) -> list[AgentTrajectory]:
    schema = _read_object(project_root / "specs/schemas/agent_trajectory.schema.json")
    trajectories: list[AgentTrajectory] = []
    for raw in read_jsonl(run_dir / "trajectories.jsonl"):
        jsonschema.Draft202012Validator(schema).validate(raw)
        trajectories.append(AgentTrajectory.model_validate(raw, strict=True))
    return trajectories


def validate_m8_reference_agent(evidence_root: Path, project_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    project_root = project_root.resolve()
    doctor = _validate_doctor(evidence_root, project_root)
    _require(doctor["docker"]["daemon_reachable"] is True, "Docker daemon was unreachable")
    checksums = _verify_checksums(evidence_root)
    tasks = load_task_suite(project_root / "tasks/mini_swe_v1", split="smoke")
    task_by_id = {task.card.id: task for task in tasks}
    expected_image = (
        ConfigRepository(project_root / "configs").experiment("mini_swe_rollout").environment.image
    )

    oracle_dir = evidence_root / "runs/oracle-docker"
    oracle_manifest = _validate_run_manifest(oracle_dir, project_root, "oracle")
    oracle_summary = AgentRunSummary.model_validate(
        _read_object(oracle_dir / "summary.json"), strict=True
    )
    _require(oracle_summary.representative is True, "oracle run was not representative")
    _require(
        oracle_summary.solved == oracle_summary.tasks == len(tasks), "oracle did not solve all"
    )
    _require(oracle_summary.mean_score == 1, "oracle mean score differs")
    _require(oracle_summary.policy_violations == 0, "oracle triggered a policy violation")
    _require(oracle_manifest["agent_environment"]["image"] == expected_image, "image differs")
    _require(oracle_manifest["agent_environment"]["replay_checked"] is True, "replay missing")
    oracle_trajectories = _trajectories(oracle_dir, project_root)
    _require(len(oracle_trajectories) == len(tasks), "oracle trajectory count differs")
    for trajectory in oracle_trajectories:
        task = task_by_id[trajectory.task_id]
        _require(
            trajectory.initial_snapshot_sha256 == task.card.snapshot_sha256,
            "trajectory reset hash differs",
        )
        _require(trajectory.finish_reason == "model_finish", "oracle did not finish normally")
        _require(trajectory.verification.public.status == "passed", "public oracle tests failed")
        _require(trajectory.verification.hidden.status == "passed", "hidden oracle tests failed")
        _require(trajectory.verification.hidden.output is None, "hidden output leaked")
        _require(trajectory.verification.final_score == 1, "oracle final score differs")
        _require(
            not any(step.policy_violations for step in trajectory.steps),
            "oracle trajectory contains policy violations",
        )
        serialized = trajectory.model_dump_json()
        _require(".nanopt_hidden_tests" not in serialized, "hidden path leaked into trajectory")
    replay = _read_object(oracle_dir / "replay.json")
    _require(set(replay) == set(task_by_id), "replay task set differs")
    _require(
        all(value["exact_semantic_match"] is True for value in replay.values()),
        "trajectory replay differed",
    )

    baseline_dir = evidence_root / "runs/model-baseline"
    baseline_manifest = _validate_run_manifest(baseline_dir, project_root, "model")
    baseline_summary = AgentRunSummary.model_validate(
        _read_object(baseline_dir / "summary.json"), strict=True
    )
    _require(baseline_summary.representative is False, "capped model baseline mislabeled")
    _require(baseline_summary.tasks == 1, "model baseline task count differs")
    baseline_trajectories = _trajectories(baseline_dir, project_root)
    _require(len(baseline_trajectories) == 1, "model baseline trajectory is missing")
    baseline = baseline_trajectories[0]
    _require(baseline.policy.name == "qwen_structured_action", "model adapter identity differs")
    _require(bool(baseline.steps), "model baseline captured no turns")
    _require(
        all(step.model_token_ids is not None for step in baseline.steps),
        "model baseline omitted exact response token IDs",
    )
    _require(baseline.verification.hidden.output is None, "baseline hidden output leaked")
    _require(
        baseline_manifest["agent_environment"]["replay_checked"] is False, "model replay mislabeled"
    )

    security = _read_object(evidence_root / "security_probes.json")
    _require(security["status"] == "passed", "Docker security probes failed")
    _require(security["image"] == expected_image, "security-probe image differs")
    probe = security["probe"]
    _require(probe["uid"] == 65532 and probe["gid"] == 65532, "container ran as root")
    _require(probe["network_blocked"] is True, "container network was reachable")
    _require(probe["root_write_blocked"] is True, "container root filesystem was writable")
    _require(probe["workspace_write_ok"] is True, "workspace mount was not writable")
    _require(probe["gpu_devices"] == [], "container exposed GPU devices")
    _require(probe["docker_socket_present"] is False, "container exposed Docker socket")
    _require(probe["cap_eff"] == "0000000000000000", "container retained capabilities")
    _require(probe["no_new_privs"] == "1", "container lacked no-new-privileges")

    return {
        "schema_version": 1,
        "status": "m8_reference_agent_passed",
        "git_commit": collect_git_metadata(project_root)["commit"],
        "hardware": {
            "profile": "rtx_4070_ti_super_16gb",
            "gpu": doctor["cuda"]["gpus"][0]["name"],
            "docker_version": doctor["docker"]["version"],
        },
        "task_suite": {
            "id": "mini_swe_v1",
            "tasks": len(tasks),
            "oracle_solved": oracle_summary.solved,
            "exact_replays": len(replay),
        },
        "oracle": {
            "run_manifest_sha256": sha256_file(oracle_dir / "run_manifest.json"),
            "trajectories_sha256": sha256_file(oracle_dir / "trajectories.jsonl"),
            "mean_score": oracle_summary.mean_score,
            "wall_seconds": oracle_summary.wall_seconds,
        },
        "model_baseline": {
            "run_manifest_sha256": sha256_file(baseline_dir / "run_manifest.json"),
            "trajectory_sha256": sha256_file(baseline_dir / "trajectories.jsonl"),
            "finish_reason": baseline.finish_reason,
            "steps": len(baseline.steps),
            "score": baseline.verification.final_score,
            "policy_violations": baseline_summary.policy_violations,
        },
        "sandbox": {
            "image": expected_image,
            "security_probes_sha256": sha256_file(evidence_root / "security_probes.json"),
            "non_root": True,
            "network_none": True,
            "gpu_hidden": True,
            "capabilities_dropped": True,
            "root_read_only": True,
            "separate_hidden_workspace": True,
        },
        "checksums": {
            "files": len(checksums),
            "manifest_sha256": sha256_file(evidence_root / "checksums.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = validate_m8_reference_agent(args.evidence_root, args.project_root)
    if args.output:
        write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
