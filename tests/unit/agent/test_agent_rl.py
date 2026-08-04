from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanopt.agent.rl_records import AgentRlAction, AgentRlEpisode, AgentRlGroup
from nanopt.agent.rl_rollout import agent_rl_seed
from nanopt.agent.rl_trainer import (
    build_agent_rl_optimizer,
    build_credit_assignment_study,
    collate_agent_rl_actions,
    update_agent_rl_policy,
)
from nanopt.config.loader import ConfigRepository
from nanopt.config.models import AgentRlExperiment


class TinyAgentRlModel(nn.Module):
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


def _action(token: int, advantage: float, *, turn: int = 0, tool: str = "finish") -> AgentRlAction:
    uniform_logp = -math.log(8)
    return AgentRlAction(
        turn_index=turn,
        prompt_token_ids=[1, 2],
        sampled_token_ids=[token],
        action_mask=[True],
        old_logprobs=[uniform_logp],
        reference_logprobs=[uniform_logp],
        decoded_text="{}",
        action_parse_status="valid",
        tool=tool,
        advantage=advantage,
    )


def _episode(index: int, token: int, advantage: float) -> AgentRlEpisode:
    return AgentRlEpisode(
        episode_id=f"episode-{index}",
        run_id="run",
        iteration=0,
        collected_policy_version=0,
        task_id="task",
        task_version="1",
        snapshot_sha256="a" * 64,
        rollout_index=index,
        actions=[_action(token, advantage)],
        finish_reason="model_finish",
        hidden_outcome_reward=float(advantage > 0),
        hidden_passed=int(advantage > 0),
        hidden_total=1,
        public_passed=advantage > 0,
        policy_violations=0,
        advantage=advantage,
    )


def _group() -> AgentRlGroup:
    return AgentRlGroup(
        group_id="group",
        run_id="run",
        iteration=0,
        policy_version=0,
        task_id="task",
        snapshot_sha256="a" * 64,
        reward_mean=0.5,
        reward_std=0.5,
        advantage_mode="group_zscore",
        degenerate=False,
        episodes=[_episode(0, 3, 1.0), _episode(1, 4, -1.0)],
    )


def _experiment(repository: ConfigRepository) -> AgentRlExperiment:
    experiment = repository.experiment("agent_rl")
    assert isinstance(experiment, AgentRlExperiment)
    optimization = experiment.optimization.model_copy(
        update={
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.2,
            "kl_beta": 0.0,
            "max_grad_norm": 10.0,
        }
    )
    rollout = experiment.rollout.model_copy(update={"iterations": 2})
    return experiment.model_copy(update={"optimization": optimization, "rollout": rollout})


def test_agent_rl_seed_uses_every_rollout_coordinate() -> None:
    base = agent_rl_seed(73, 0, "task", 0, 0)
    assert base == agent_rl_seed(73, 0, "task", 0, 0)
    assert (
        len(
            {
                base,
                agent_rl_seed(73, 1, "task", 0, 0),
                agent_rl_seed(73, 0, "other", 0, 0),
                agent_rl_seed(73, 0, "task", 1, 0),
                agent_rl_seed(73, 0, "task", 0, 1),
            }
        )
        == 5
    )


def test_agent_rl_group_rejects_mixed_snapshots() -> None:
    value = _group().model_dump(mode="python")
    value["episodes"][1]["snapshot_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="share task and initial snapshot"):
        AgentRlGroup.model_validate(value, strict=True)


def test_collator_preserves_prompt_boundary_and_old_logp_coordinate() -> None:
    batch = collate_agent_rl_actions([_action(3, 1.0), _action(4, -1.0)], pad_token_id=0)
    assert batch.input_ids.tolist() == [[1, 2, 3], [1, 2, 4]]
    assert batch.action_mask.tolist() == [[False, False, True], [False, False, True]]
    assert batch.old_logprobs[:, 0].tolist() == [0.0, 0.0]
    assert batch.old_logprobs[0, 1].item() == pytest.approx(-math.log(8))


def test_tiny_agent_rl_update_favors_the_successful_action(
    config_repository: ConfigRepository,
) -> None:
    model = TinyAgentRlModel()
    experiment = _experiment(config_repository)
    optimizer = build_agent_rl_optimizer(model, experiment)

    metrics = update_agent_rl_policy(
        model,
        [_group()],
        experiment,
        optimizer,
        iteration=0,
        policy_version=0,
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    assert metrics.optimizer_steps == 1
    assert model.lora_weight[2, 3] > model.lora_weight[2, 4]


def test_agent_rl_update_rejects_stale_policy_versions(
    config_repository: ConfigRepository,
) -> None:
    model = TinyAgentRlModel()
    experiment = _experiment(config_repository)
    optimizer = build_agent_rl_optimizer(model, experiment)
    with pytest.raises(ValueError, match="refuses stale"):
        update_agent_rl_policy(
            model,
            [_group()],
            experiment,
            optimizer,
            iteration=1,
            policy_version=1,
            pad_token_id=0,
            device=torch.device("cpu"),
        )


def test_credit_study_makes_terminal_only_coverage_visible() -> None:
    group = _group()
    group.episodes[0].actions = [
        _action(5, 1.0, turn=0, tool="read_file"),
        _action(3, 1.0, turn=1, tool="finish"),
    ]
    study = build_credit_assignment_study([group])
    assert study.all_actions_active_tokens == 3
    assert study.terminal_action_active_tokens == 2
    assert study.active_tokens_by_tool == {"finish": 2, "read_file": 1}
