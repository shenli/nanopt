from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from nanopt.config.loader import ConfigRepository
from nanopt.config.models import GrpoExperiment
from nanopt.data.arithmetic import generate_task
from nanopt.grpo.rollout import (
    deterministic_prompt_schedule,
    generate_grouped_trajectory,
    rollout_seed,
)
from nanopt.models.renderer import ChatRenderer


class TinyTokenizer:
    chat_template = "tiny"

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_tensors: str,
    ) -> list[int]:
        del conversation, tokenize, add_generation_prompt, enable_thinking, return_tensors
        return [1, 2]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del token_ids, skip_special_tokens
        return f"<answer>{self.answer}</answer>"


class UniformModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        del attention_mask
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 16) + self.anchor * 0)


def _experiment(config_repository: ConfigRepository) -> GrpoExperiment:
    experiment = config_repository.experiment("math_grpo")
    assert isinstance(experiment, GrpoExperiment)
    rollout = experiment.rollout.model_copy(update={"group_size": 2, "max_completion_length": 2})
    return experiment.model_copy(update={"rollout": rollout})


def test_grouped_rollout_stores_exact_ids_logps_rewards_and_zero_degenerate_advantages(
    config_repository: ConfigRepository,
) -> None:
    task = generate_task(family="addition_subtraction", difficulty=1, seed=4)
    tokenizer = TinyTokenizer(task.target.canonical_answer)
    trajectory = generate_grouped_trajectory(
        UniformModel(),
        tokenizer,
        ChatRenderer(tokenizer),
        task,
        _experiment(config_repository),
        run_id="run",
        iteration=0,
        eos_token_id=15,
        stop_token_sequence=(14, 15),
    )

    assert len(trajectory.completions) == 2
    assert trajectory.group_reward_std == 0
    assert [completion.advantage for completion in trajectory.completions] == [0, 0]
    for completion in trajectory.completions:
        assert len(completion.token_ids) == len(completion.old_logprobs)
        assert completion.action_mask == [1] * len(completion.token_ids)
        assert completion.verifier_status == "correct"


def test_prompt_and_rollout_seed_schedules_are_reproducible() -> None:
    tasks = [
        generate_task(family="addition_subtraction", difficulty=1, seed=seed) for seed in range(3)
    ]
    first = deterministic_prompt_schedule(tasks, iterations=5, batch_size=1, seed=7)
    second = deterministic_prompt_schedule(tasks, iterations=5, batch_size=1, seed=7)

    assert [[task.task_id for task in group] for group in first] == [
        [task.task_id for task in group] for group in second
    ]
    assert rollout_seed(1, 2, "task", 3) == rollout_seed(1, 2, "task", 3)
    assert rollout_seed(1, 2, "task", 3) != rollout_seed(1, 2, "task", 4)
