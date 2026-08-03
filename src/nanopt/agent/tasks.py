"""Load original MiniSWE task cards while keeping hidden assets outside model workspaces."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from nanopt.agent.records import AgentSuite, AgentTaskCard
from nanopt.sft.checkpoint import sha256_directory


@dataclass(frozen=True)
class LoadedAgentTask:
    """Trusted task metadata plus source paths never serialized into observations."""

    card: AgentTaskCard
    task_dir: Path
    snapshot_dir: Path
    hidden_tests_dir: Path
    oracle_patch_path: Path


def _load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid task YAML {path}: {exc}") from exc


def _require_plain_tree(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{label} directory is missing: {path}")
    files = [entry for entry in path.rglob("*") if entry.is_file()]
    if not files:
        raise ValueError(f"{label} directory is empty: {path}")
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"{label} contains a symbolic link: {entry.relative_to(path)}")


def load_agent_task(task_dir: Path) -> LoadedAgentTask:
    """Load one task and prove its immutable snapshot hash before any episode starts."""

    card = AgentTaskCard.model_validate(_load_yaml(task_dir / "task.yaml"), strict=True)
    if task_dir.name != card.id:
        raise ValueError(f"task directory {task_dir.name!r} differs from card ID {card.id!r}")
    snapshot = task_dir / "snapshot"
    hidden = task_dir / "hidden_tests"
    oracle = task_dir / "oracle.patch"
    _require_plain_tree(snapshot, "snapshot")
    _require_plain_tree(hidden, "hidden tests")
    if not oracle.is_file() or oracle.is_symlink():
        raise ValueError(f"oracle patch is missing or unsafe: {oracle}")
    actual = sha256_directory(snapshot)
    if actual != card.snapshot_sha256:
        raise ValueError(
            f"task {card.id} snapshot hash differs: expected {card.snapshot_sha256}, got {actual}"
        )
    return LoadedAgentTask(card, task_dir, snapshot, hidden, oracle)


def load_task_suite(root: Path, *, split: str | None = None) -> list[LoadedAgentTask]:
    """Load a deterministic suite order and reject duplicate or unlisted task identities."""

    suite = AgentSuite.model_validate(_load_yaml(root / "suite.yaml"), strict=True)
    if len(set(suite.tasks)) != len(suite.tasks):
        raise ValueError("agent suite contains duplicate task IDs")
    tasks = [load_agent_task(root / task_id) for task_id in suite.tasks]
    if split is not None:
        if split not in {"smoke", "reference", "all"}:
            raise ValueError("task split must be smoke, reference, or all")
        if split != "all":
            tasks = [task for task in tasks if task.card.split == split]
    if not tasks:
        raise ValueError("selected agent task suite is empty")
    return tasks


def copy_snapshot(task: LoadedAgentTask, destination: Path) -> str:
    """Create a new episode workspace without copying task metadata or hidden tests."""

    if destination.exists():
        raise ValueError(f"episode workspace already exists: {destination}")
    shutil.copytree(task.snapshot_dir, destination, symlinks=False)
    copied_hash = sha256_directory(destination)
    if copied_hash != task.card.snapshot_sha256:
        raise RuntimeError(f"copied workspace hash differs for task {task.card.id}")
    return copied_hash
