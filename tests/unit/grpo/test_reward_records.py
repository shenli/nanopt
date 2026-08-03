from __future__ import annotations

import json

from nanopt.config.loader import ConfigRepository
from nanopt.config.models import GrpoExperiment
from nanopt.data.arithmetic import generate_task
from nanopt.grpo.records import (
    GrpoCompletionRecord,
    GrpoPromptRecord,
    GrpoTrajectoryRecord,
)
from nanopt.grpo.reward import arithmetic_rlvr_reward, reward_hacking_suite


def _experiment(config_repository: ConfigRepository) -> GrpoExperiment:
    experiment = config_repository.experiment("math_grpo")
    assert isinstance(experiment, GrpoExperiment)
    return experiment


def test_arithmetic_reward_separates_parser_and_verifier_failures(
    config_repository: ConfigRepository,
) -> None:
    task = generate_task(family="addition_subtraction", difficulty=1, seed=9)
    weights = _experiment(config_repository).reward.components
    correct = arithmetic_rlvr_reward(
        task,
        f"<answer>{task.target.canonical_answer}</answer>",
        weights,
        completion_tokens=3,
    )
    invalid = arithmetic_rlvr_reward(
        task,
        f"<answer>{task.target.canonical_answer}</answer> trailing",
        weights,
        completion_tokens=4,
    )

    assert correct.reward == 1.1
    assert correct.parser_status == "valid"
    assert correct.verifier_status == "correct"
    assert invalid.reward == 0
    assert invalid.parser_status == "invalid"
    assert invalid.verifier_status == "not_run"
    assert all(result["passed"] is True for result in reward_hacking_suite(task, weights))


def test_trajectory_json_round_trip_preserves_exact_token_coordinates() -> None:
    completion = GrpoCompletionRecord(
        completion_index=0,
        token_ids=[7, 8],
        action_mask=[1, 1],
        old_logprobs=[-0.5, -0.25],
        decoded_text="<answer>1</answer>",
        finish_reason="protocol_stop",
        reward=1.1,
        reward_components={"correctness_reward": 1.0},
        advantage=1.0,
        parser_status="valid",
        parsed_answer="1",
        verifier_status="correct",
        generation_seconds=0.01,
    )
    trajectory = GrpoTrajectoryRecord(
        trajectory_id="trajectory",
        run_id="run",
        iteration=0,
        task_id="task",
        prompt=GrpoPromptRecord(
            messages=[{"role": "user", "content": "x"}],
            token_ids=[1, 2],
            attention_mask=[1, 1],
        ),
        group_reward_mean=0.55,
        group_reward_std=0.55,
        advantage_mode="group_zscore",
        completions=[completion, completion.model_copy(update={"completion_index": 1})],
    )

    restored = GrpoTrajectoryRecord.model_validate_json(
        json.dumps(trajectory.model_dump(mode="json")), strict=True
    )

    assert restored == trajectory
    assert restored.completions[0].token_ids == [7, 8]
    assert restored.completions[0].old_logprobs == [-0.5, -0.25]
