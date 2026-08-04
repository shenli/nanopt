"""Inspect identical-snapshot advantages and action-token credit without loading a model."""

from __future__ import annotations

import torch

from nanopt.agent.rl_records import AgentRlAction, AgentRlEpisode, AgentRlGroup
from nanopt.agent.rl_trainer import build_credit_assignment_study
from nanopt.core.advantages import group_relative_advantages


def action(turn: int, tool: str, token_count: int, advantage: float) -> AgentRlAction:
    return AgentRlAction(
        turn_index=turn,
        prompt_token_ids=[10, 11],
        sampled_token_ids=list(range(20, 20 + token_count)),
        action_mask=[True] * token_count,
        old_logprobs=[-1.0] * token_count,
        decoded_text="{}",
        action_parse_status="valid",
        tool=tool,
        advantage=advantage,
    )


def episode(index: int, reward: float, advantage: float) -> AgentRlEpisode:
    return AgentRlEpisode(
        episode_id=f"episode-{index}",
        run_id="cpu-lab",
        iteration=0,
        collected_policy_version=0,
        task_id="same-task",
        task_version="1",
        snapshot_sha256="a" * 64,
        rollout_index=index,
        actions=[
            action(0, "read_file", 2, advantage),
            action(1, "finish", 1, advantage),
        ],
        finish_reason="model_finish",
        hidden_outcome_reward=reward,
        hidden_passed=int(reward),
        hidden_total=1,
        public_passed=bool(reward),
        policy_violations=0,
        advantage=advantage,
    )


def main() -> None:
    rewards = torch.tensor([[1.0, 0.0]])
    relative = group_relative_advantages(rewards, mode="group_zscore", epsilon=1e-4)
    advantages = relative.advantages[0].tolist()
    group = AgentRlGroup(
        group_id="identical-snapshot-group",
        run_id="cpu-lab",
        iteration=0,
        policy_version=0,
        task_id="same-task",
        snapshot_sha256="a" * 64,
        reward_mean=float(relative.group_mean[0]),
        reward_std=float(relative.group_std[0]),
        advantage_mode="group_zscore",
        degenerate=False,
        episodes=[episode(0, 1.0, advantages[0]), episode(1, 0.0, advantages[1])],
    )
    study = build_credit_assignment_study([group])

    assert group.episodes[0].snapshot_sha256 == group.episodes[1].snapshot_sha256
    assert advantages[0] > 0 > advantages[1]
    assert study.all_actions_active_tokens == 6
    assert study.terminal_action_active_tokens == 2
    print("Agent RL identical-snapshot and credit-assignment invariants passed")


if __name__ == "__main__":
    main()
