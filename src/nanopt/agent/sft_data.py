"""Collect trusted tool trajectories and freeze their exact Agent SFT token targets."""

from __future__ import annotations

from pathlib import Path

from nanopt.agent.context import trajectory_messages
from nanopt.agent.environment import MiniSWEEnvironment, trajectory_semantics
from nanopt.agent.policy import RecoveryOraclePolicy, ReplayPolicy, ScriptedOraclePolicy
from nanopt.agent.records import AgentTrajectory
from nanopt.agent.sandbox import FakeSandboxBackend, SandboxLimits
from nanopt.agent.sft_records import (
    AgentChatMessage,
    AgentSftDatasetManifest,
    AgentSftExample,
)
from nanopt.agent.tasks import LoadedAgentTask, load_task_suite
from nanopt.config.models import AgentSftExperiment, ModelProfile
from nanopt.models.loading import load_qwen_tokenizer, qwen_chat_terminator_id
from nanopt.models.renderer import ChatRenderer, RenderedSupervisedExample
from nanopt.runtime.artifacts import (
    append_jsonl,
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from nanopt.sft.checkpoint import sha256_directory

TOOLS = ["list_files", "read_file", "search", "apply_patch", "run_tests", "finish"]


def _collect(
    task: LoadedAgentTask,
    *,
    policy: ScriptedOraclePolicy,
    run_id: str,
) -> AgentTrajectory:
    limits = SandboxLimits(timeout_seconds=10, memory_mib=256, pids=32)
    with MiniSWEEnvironment(
        task,
        FakeSandboxBackend(),
        run_id=run_id,
        allowed_tools=TOOLS,  # type: ignore[arg-type]
        limits=limits,
    ) as environment:
        trajectory = environment.run_episode(policy)
    if trajectory.verification.public.status != "passed":
        raise ValueError(f"trusted trajectory failed public tests for {task.card.id}")
    if trajectory.verification.hidden.status != "passed":
        raise ValueError(f"trusted trajectory failed hidden tests for {task.card.id}")

    replay = ReplayPolicy([step.model_response for step in trajectory.steps], trajectory.policy)
    with MiniSWEEnvironment(
        task,
        FakeSandboxBackend(),
        run_id=run_id,
        allowed_tools=TOOLS,  # type: ignore[arg-type]
        limits=limits,
    ) as environment:
        repeated = environment.run_episode(replay)
    if trajectory_semantics(trajectory) != trajectory_semantics(repeated):
        raise RuntimeError(f"exact semantic replay failed for {trajectory.trajectory_id}")
    return trajectory


def _render_example(
    trajectory: AgentTrajectory,
    *,
    step_index: int,
    split: str,
    kind: str,
    renderer: ChatRenderer,
    context_policy: str,
    source_sha256: str,
) -> AgentSftExample:
    step = trajectory.steps[step_index]
    if step.action_parse_status != "valid" or step.action is None:
        raise ValueError("Agent SFT targets must be valid typed actions")
    messages = trajectory_messages(
        trajectory,
        step_index,
        context_policy=context_policy,  # type: ignore[arg-type]
    )
    rendered = renderer.render_supervised(messages, step.model_response)
    identity = canonical_json(
        {
            "trajectory_id": trajectory.trajectory_id,
            "step_index": step_index,
            "context_policy": context_policy,
            "kind": kind,
            "source_sha256": source_sha256,
        }
    )
    return AgentSftExample(
        example_id=f"agent_sft_{sha256_bytes(identity)[:24]}",
        split=split,  # type: ignore[arg-type]
        example_kind=kind,  # type: ignore[arg-type]
        task_id=trajectory.task_id,
        task_version=trajectory.task_version,
        trajectory_id=trajectory.trajectory_id,
        source_trajectory_sha256=source_sha256,
        step_index=step_index,
        context_policy=context_policy,  # type: ignore[arg-type]
        messages=[AgentChatMessage.model_validate(item, strict=True) for item in messages],
        completion=step.model_response,
        target_action=step.action,
        input_ids=list(rendered.input_ids),
        attention_mask=list(rendered.attention_mask),
        action_mask=list(rendered.action_mask),
        prompt_length=rendered.prompt_length,
        chat_template_sha256=rendered.chat_template_sha256,
    )


def build_agent_sft_dataset(
    experiment: AgentSftExperiment,
    model_profile: ModelProfile,
    *,
    tasks_root: Path,
    output_dir: Path,
    local_files_only: bool,
) -> AgentSftDatasetManifest:
    """Create a new immutable-style dataset directory from replay-checked oracle episodes."""

    if output_dir.exists():
        raise ValueError("Agent SFT output directory must not already exist")
    tasks = load_task_suite(tasks_root, split="all")
    by_id = {task.card.id: task for task in tasks}
    selected_ids = [*experiment.data.train_tasks, *experiment.data.validation_tasks]
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise ValueError(f"Agent SFT profile names unknown tasks: {', '.join(missing)}")

    tokenizer = load_qwen_tokenizer(model_profile, local_files_only=local_files_only)
    renderer = ChatRenderer(
        tokenizer,
        enable_thinking=model_profile.renderer.enable_thinking,
        terminal_token_id=qwen_chat_terminator_id(tokenizer),
    )
    output_dir.mkdir(parents=True)
    trajectory_dir = output_dir / "source_trajectories"
    trajectory_dir.mkdir()
    examples_path = output_dir / "examples.jsonl"
    examples: list[AgentSftExample] = []
    trajectories: list[AgentTrajectory] = []

    for task_id in selected_ids:
        task = by_id[task_id]
        split = "train" if task_id in experiment.data.train_tasks else "validation"
        patch = task.oracle_patch_path.read_text(encoding="utf-8")
        demonstration = _collect(
            task,
            policy=ScriptedOraclePolicy(patch),
            run_id=f"agent-sft-{task_id}-demonstration",
        )
        trajectories.append(demonstration)
        demo_path = trajectory_dir / f"{task_id}-demonstration.json"
        write_json(demo_path, demonstration.model_dump(mode="json"))
        demo_sha = sha256_file(demo_path)
        for step_index in range(len(demonstration.steps)):
            examples.append(
                _render_example(
                    demonstration,
                    step_index=step_index,
                    split=split,
                    kind="demonstration",
                    renderer=renderer,
                    context_policy=experiment.data.context_policy,
                    source_sha256=demo_sha,
                )
            )

        if experiment.data.include_recovery_examples:
            recovery = _collect(
                task,
                policy=RecoveryOraclePolicy(patch),
                run_id=f"agent-sft-{task_id}-recovery",
            )
            trajectories.append(recovery)
            recovery_path = trajectory_dir / f"{task_id}-recovery.json"
            write_json(recovery_path, recovery.model_dump(mode="json"))
            recovery_sha = sha256_file(recovery_path)
            recovery_index = next(
                index + 1
                for index, step in enumerate(recovery.steps[:-1])
                if step.action_parse_status == "invalid"
            )
            examples.append(
                _render_example(
                    recovery,
                    step_index=recovery_index,
                    split=split,
                    kind="recovery",
                    renderer=renderer,
                    context_policy=experiment.data.context_policy,
                    source_sha256=recovery_sha,
                )
            )

    for example in examples:
        append_jsonl(examples_path, example.model_dump(mode="json"))
    examples_sha = sha256_file(examples_path)
    trajectories_sha = sha256_directory(trajectory_dir)
    dataset_sha = sha256_bytes(
        canonical_json(
            {
                "examples_sha256": examples_sha,
                "source_trajectories_sha256": trajectories_sha,
                "context_policy": experiment.data.context_policy,
            }
        )
    )
    manifest = AgentSftDatasetManifest(
        dataset_id=experiment.data.dataset,
        dataset_sha256=dataset_sha,
        suite_id=tasks_root.name,
        suite_version="1.0.0",
        context_policy=experiment.data.context_policy,
        tokenizer_revision=model_profile.source.tokenizer_revision
        or model_profile.source.revision
        or "unresolved",
        chat_template_sha256=renderer.chat_template_sha256,
        examples_file=examples_path.name,
        examples_sha256=examples_sha,
        source_trajectories_sha256=trajectories_sha,
        train_tasks=experiment.data.train_tasks,
        validation_tasks=experiment.data.validation_tasks,
        train_examples=sum(item.split == "train" for item in examples),
        validation_examples=sum(item.split == "validation" for item in examples),
        demonstration_examples=sum(item.example_kind == "demonstration" for item in examples),
        recovery_examples=sum(item.example_kind == "recovery" for item in examples),
        exact_replays_passed=len(trajectories),
        source_trajectories=len(trajectories),
    )
    write_json(output_dir / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def read_agent_sft_dataset(
    dataset_dir: Path,
) -> tuple[AgentSftDatasetManifest, list[AgentSftExample]]:
    """Validate a frozen dataset and return records without rendering them again."""

    manifest = AgentSftDatasetManifest.model_validate_json(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8"), strict=True
    )
    examples_path = dataset_dir / manifest.examples_file
    if sha256_file(examples_path) != manifest.examples_sha256:
        raise ValueError("Agent SFT examples hash differs from the manifest")
    if sha256_directory(dataset_dir / "source_trajectories") != (
        manifest.source_trajectories_sha256
    ):
        raise ValueError("Agent SFT source-trajectory hash differs from the manifest")
    examples = [
        AgentSftExample.model_validate_json(line, strict=True)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(examples) != manifest.train_examples + manifest.validation_examples:
        raise ValueError("Agent SFT example count differs from the manifest")
    sources = {
        sha256_file(path): AgentTrajectory.model_validate_json(
            path.read_text(encoding="utf-8"), strict=True
        )
        for path in sorted((dataset_dir / "source_trajectories").glob("*.json"))
    }
    for example in examples:
        source = sources.get(example.source_trajectory_sha256)
        if source is None or source.trajectory_id != example.trajectory_id:
            raise ValueError(
                f"Agent SFT example {example.example_id} has broken trajectory lineage"
            )
    return manifest, examples


def stored_rendered_example(example: AgentSftExample) -> RenderedSupervisedExample:
    """Adapt a validated record to the generic trainer without decoding or re-tokenizing."""

    return RenderedSupervisedExample(
        input_ids=tuple(example.input_ids),
        attention_mask=tuple(example.attention_mask),
        action_mask=tuple(example.action_mask),
        prompt_length=example.prompt_length,
        chat_template_sha256=example.chat_template_sha256,
    )
