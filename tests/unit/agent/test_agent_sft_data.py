from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nanopt.agent.context import trajectory_messages
from nanopt.agent.policy import RecoveryOraclePolicy, ScriptedOraclePolicy
from nanopt.agent.sft_data import _collect, _render_example, stored_rendered_example
from nanopt.agent.sft_records import AgentSftExample
from nanopt.agent.tasks import load_task_suite
from nanopt.models.renderer import ChatRenderer


class FakeChatTokenizer:
    chat_template = "<role>{{ role }}</role>{{ content }}"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_tensors: str,
    ) -> Any:
        del tokenize, enable_thinking, return_tensors
        roles = {"system": 10, "user": 20, "assistant": 30}
        ids = [1]
        for message in conversation:
            ids.extend([roles[message["role"]], *message["content"].encode(), 2])
        if add_generation_prompt:
            ids.append(roles["assistant"])
        return ids


def _task_and_patch(project_root: Path) -> tuple[Any, str]:
    task = load_task_suite(project_root / "tasks/mini_swe_v1", split="all")[0]
    return task, task.oracle_patch_path.read_text(encoding="utf-8")


def test_oracle_demonstration_covers_inspection_edit_test_and_finish(project_root: Path) -> None:
    task, patch = _task_and_patch(project_root)
    trajectory = _collect(
        task,
        policy=ScriptedOraclePolicy(patch),
        run_id="agent-sft-demonstration-test",
    )

    assert [step.action["tool"] for step in trajectory.steps if step.action] == [
        "list_files",
        "read_file",
        "apply_patch",
        "run_tests",
        "finish",
    ]
    assert trajectory.verification.final_score == 1.0


def test_full_transcript_preserves_raw_recovery_turn_and_exact_action_mask(
    project_root: Path,
) -> None:
    task, patch = _task_and_patch(project_root)
    trajectory = _collect(
        task,
        policy=RecoveryOraclePolicy(patch),
        run_id="agent-sft-recovery-test",
    )
    messages = trajectory_messages(trajectory, 1, context_policy="full_transcript")

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == "this is not a JSON action"
    assert '"code":"invalid_action"' in messages[3]["content"]
    assert '"transcript":[]' in messages[3]["content"]

    renderer = ChatRenderer(FakeChatTokenizer())
    example = _render_example(
        trajectory,
        step_index=1,
        split="train",
        kind="recovery",
        renderer=renderer,
        context_policy="full_transcript",
        source_sha256="a" * 64,
    )
    stored = stored_rendered_example(example)

    assert example.example_kind == "recovery"
    assert example.target_action["tool"] == "list_files"
    assert not any(example.action_mask[: example.prompt_length])
    assert all(example.action_mask[example.prompt_length :])
    assert stored.input_ids == tuple(example.input_ids)


def test_agent_sft_record_rejects_mask_coordinate_drift(project_root: Path) -> None:
    task, patch = _task_and_patch(project_root)
    trajectory = _collect(
        task,
        policy=ScriptedOraclePolicy(patch),
        run_id="agent-sft-mask-test",
    )
    example = _render_example(
        trajectory,
        step_index=0,
        split="train",
        kind="demonstration",
        renderer=ChatRenderer(FakeChatTokenizer()),
        context_policy="observation_snapshot",
        source_sha256="b" * 64,
    )
    value = example.model_dump(mode="python")
    value["action_mask"] = value["action_mask"][:-1]

    with pytest.raises(ValueError, match="same full-sequence coordinates"):
        AgentSftExample.model_validate(value, strict=True)
