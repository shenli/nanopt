from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from nanopt.config.loader import ConfigRepository
from nanopt.config.models import GrpoExperiment
from nanopt.grpo.records import (
    GrpoCompletionRecord,
    GrpoPromptRecord,
    GrpoTrajectoryRecord,
)
from nanopt.grpo.trainer import (
    build_grpo_optimizer,
    collate_grpo_completions,
    update_grpo_policy,
)


class TinyGrpoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_weight = nn.Parameter(torch.zeros(8, 8))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.lora_weight[input_ids])


def _trajectory() -> GrpoTrajectoryRecord:
    prompt = GrpoPromptRecord(
        messages=[{"role": "user", "content": "x"}],
        token_ids=[1, 2],
        attention_mask=[1, 1],
    )

    def completion(index: int, token: int, advantage: float) -> GrpoCompletionRecord:
        return GrpoCompletionRecord(
            completion_index=index,
            token_ids=[token],
            action_mask=[1],
            old_logprobs=[-2.0794415],
            decoded_text=str(token),
            finish_reason="max_length",
            reward=float(advantage > 0),
            reward_components={"correctness_reward": float(advantage > 0)},
            advantage=advantage,
            parser_status="valid",
            parsed_answer=str(token),
            verifier_status="correct" if advantage > 0 else "incorrect",
            generation_seconds=0,
        )

    return GrpoTrajectoryRecord(
        trajectory_id="trajectory",
        run_id="run",
        iteration=0,
        task_id="task",
        prompt=prompt,
        group_reward_mean=0.5,
        group_reward_std=0.5,
        advantage_mode="group_zscore",
        completions=[completion(0, 3, 1.0), completion(1, 4, -1.0)],
    )


def _experiment(config_repository: ConfigRepository) -> GrpoExperiment:
    experiment = config_repository.experiment("math_grpo")
    assert isinstance(experiment, GrpoExperiment)
    optimization = experiment.optimization.model_copy(
        update={
            "iterations": 4,
            "minibatch_completions": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.2,
            "warmup_ratio": 0.0,
            "max_grad_norm": 10.0,
        }
    )
    return experiment.model_copy(update={"optimization": optimization})


def test_collator_preserves_stored_ids_and_causal_old_logp_coordinates() -> None:
    trajectory = _trajectory()
    values = [(trajectory, completion) for completion in trajectory.completions]

    batch = collate_grpo_completions(values, pad_token_id=0)

    assert batch.input_ids.tolist() == [[1, 2, 3], [1, 2, 4]]
    assert batch.action_mask.tolist() == [[False, False, True], [False, False, True]]
    assert batch.old_logprobs.tolist()[0] == [0.0, -2.079441547393799]


def test_tiny_grpo_update_increases_higher_reward_token_probability(
    config_repository: ConfigRepository,
) -> None:
    model = TinyGrpoModel()
    experiment = _experiment(config_repository)
    trajectory = _trajectory()
    optimizer = build_grpo_optimizer(model, experiment)

    metrics = update_grpo_policy(
        model,
        [trajectory],
        experiment,
        optimizer,
        iteration=0,
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    assert metrics.optimizer_steps == 1
    assert model.lora_weight[2, 3] > model.lora_weight[2, 4]
