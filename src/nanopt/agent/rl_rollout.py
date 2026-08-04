"""Grouped stateful Agent RL rollouts with exact online token evidence."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import torch

from nanopt.agent.context import agent_system_instruction, observation_text
from nanopt.agent.environment import MiniSWEEnvironment
from nanopt.agent.policy import PolicyResponse
from nanopt.agent.records import AgentObservation, AgentPolicyIdentity, parse_action
from nanopt.agent.rl_records import AgentRlAction, AgentRlEpisode, AgentRlGroup
from nanopt.agent.sandbox.base import SandboxBackend, SandboxLimits
from nanopt.agent.tasks import LoadedAgentTask
from nanopt.config.models import AgentRlExperiment
from nanopt.core.advantages import group_relative_advantages
from nanopt.models.renderer import ChatRenderer
from nanopt.rollout.sampler import GenerationResult, SamplingConfig, sample_autoregressive
from nanopt.runtime.artifacts import canonical_json, sha256_bytes


def agent_rl_seed(
    base_seed: int,
    iteration: int,
    task_id: str,
    rollout_index: int,
    turn_index: int,
) -> int:
    """Derive one stable private sampling seed from all rollout coordinates."""

    if iteration < 0 or rollout_index < 0 or turn_index < 0 or not task_id:
        raise ValueError("Agent RL seed coordinates are invalid")
    value = (
        f"agent-rl-rollout-v1\0{base_seed}\0{iteration}\0{task_id}\0{rollout_index}\0{turn_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


def has_unexecuted_timeout_generation(
    generation_count: int, step_count: int, finish_reason: str
) -> bool:
    """Validate generation/step alignment and identify one deadline-crossing action."""

    if generation_count == step_count:
        return False
    if generation_count == step_count + 1 and finish_reason == "timeout":
        return True
    raise RuntimeError("Agent RL generation count differs from environment step count")


class ExactAgentRolloutPolicy:
    """Generate actions while retaining the prompt and sampled distribution for every turn.

    The environment receives only decoded JSON text and sampled IDs. The richer generation result
    stays on this trusted collector, so behavior log probabilities never become an observation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        renderer: ChatRenderer,
        *,
        checkpoint_id: str,
        base_seed: int,
        iteration: int,
        task_id: str,
        rollout_index: int,
        policy_version: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.base_seed = base_seed
        self.iteration = iteration
        self.task_id = task_id
        self.rollout_index = rollout_index
        self.policy_version = policy_version
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.turn = 0
        self._conversation: list[dict[str, str]] = []
        self._previous_response: str | None = None
        self.generations: list[GenerationResult] = []
        self.identity = AgentPolicyIdentity(
            name="qwen_exact_agent_rl",
            version="1",
            checkpoint_id=checkpoint_id,
            generation={
                "policy_version": policy_version,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "context_policy": "full_transcript",
                "exact_token_ids_saved": True,
                "behavior_logprobs_saved": True,
                "hidden_reward_exposed": False,
            },
        )

    def _messages(self, observation: AgentObservation) -> list[dict[str, str]]:
        if not observation.transcript:
            self._conversation = [
                {"role": "system", "content": agent_system_instruction()},
                {
                    "role": "user",
                    "content": observation_text(observation, include_transcript=False),
                },
            ]
            self._previous_response = None
        else:
            if self._previous_response is None or not self._conversation:
                raise RuntimeError("Agent RL conversation lost its preceding sampled action")
            self._conversation.extend(
                [
                    {"role": "assistant", "content": self._previous_response},
                    {
                        "role": "user",
                        "content": observation_text(observation, include_transcript=False),
                    },
                ]
            )
            self._previous_response = None
        return list(self._conversation)

    def _is_complete_json_action(self, token_ids: tuple[int, ...]) -> bool:
        value = self.tokenizer.decode(list(token_ids), skip_special_tokens=True)
        if not isinstance(value, str):
            raise TypeError("tokenizer decode must return text")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict)

    def respond(self, observation: AgentObservation) -> PolicyResponse:
        prompt = self.renderer.render_prompt(self._messages(observation))
        seed = agent_rl_seed(
            self.base_seed,
            self.iteration,
            self.task_id,
            self.rollout_index,
            self.turn,
        )
        started = time.perf_counter()
        generation = sample_autoregressive(
            self.model,
            torch.tensor(prompt.input_ids, dtype=torch.long),
            SamplingConfig(
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.renderer.terminal_token_id,
            ),
            seed=seed,
            stop_predicate=self._is_complete_json_action,
        )
        seconds = max(0.0, time.perf_counter() - started)
        value = self.tokenizer.decode(
            list(generation.generated_token_ids), skip_special_tokens=True
        )
        if not isinstance(value, str):
            raise TypeError("tokenizer decode must return text")
        self.generations.append(generation)
        self._previous_response = value
        self.turn += 1
        return PolicyResponse(value, list(generation.generated_token_ids), seconds)


def generate_agent_rl_episode(
    model: Any,
    tokenizer: Any,
    renderer: ChatRenderer,
    task: LoadedAgentTask,
    experiment: AgentRlExperiment,
    backend: SandboxBackend,
    limits: SandboxLimits,
    *,
    run_id: str,
    iteration: int,
    rollout_index: int,
    policy_version: int,
    checkpoint_id: str,
    do_sample: bool = True,
    tool_call_limit: int | None = None,
) -> AgentRlEpisode:
    """Reset one workspace, sample a full episode, then reveal its hidden outcome reward."""

    policy = ExactAgentRolloutPolicy(
        model,
        tokenizer,
        renderer,
        checkpoint_id=checkpoint_id,
        base_seed=experiment.seed,
        iteration=iteration,
        task_id=task.card.id,
        rollout_index=rollout_index,
        policy_version=policy_version,
        max_new_tokens=experiment.rollout.max_new_tokens_per_turn,
        do_sample=do_sample,
        temperature=experiment.rollout.temperature,
        top_p=experiment.rollout.top_p,
    )
    with MiniSWEEnvironment(
        task,
        backend,
        run_id=run_id,
        allowed_tools=list(experiment.tools),
        limits=limits,
        turn_limit=experiment.rollout.max_turns,
        tool_call_limit=tool_call_limit,
    ) as environment:
        trajectory = environment.run_episode(policy)
    # A response may finish sampling just after the wall-clock deadline. The environment then
    # terminates before executing it, leaving one more policy generation than environment step.
    # Retain that sampled action: it consumed policy compute and contributed to the zero/partial
    # terminal outcome, so silently dropping it would bias the policy-gradient dataset.
    retain_timeout = has_unexecuted_timeout_generation(
        len(policy.generations), len(trajectory.steps), trajectory.finish_reason
    )
    retained_timeout_generation = policy.generations[-1] if retain_timeout else None

    actions: list[AgentRlAction] = []
    for step, generation in zip(trajectory.steps, policy.generations, strict=True):
        tool = str(step.action["tool"]) if step.action is not None else None
        actions.append(
            AgentRlAction(
                turn_index=step.step_index,
                prompt_token_ids=list(generation.prompt_token_ids),
                sampled_token_ids=list(generation.generated_token_ids),
                action_mask=list(generation.active_mask),
                old_logprobs=list(generation.behavior_logps),
                decoded_text=step.model_response,
                action_parse_status=step.action_parse_status,
                tool=tool,
            )
        )
    if retained_timeout_generation is not None:
        decoded = tokenizer.decode(
            list(retained_timeout_generation.generated_token_ids), skip_special_tokens=True
        )
        if not isinstance(decoded, str):
            raise TypeError("tokenizer decode must return text")
        try:
            timed_out_action = parse_action(decoded)
            parse_status = "valid"
            tool = timed_out_action.tool
        except ValueError:
            parse_status = "invalid"
            tool = None
        actions.append(
            AgentRlAction(
                turn_index=len(actions),
                prompt_token_ids=list(retained_timeout_generation.prompt_token_ids),
                sampled_token_ids=list(retained_timeout_generation.generated_token_ids),
                action_mask=list(retained_timeout_generation.active_mask),
                old_logprobs=list(retained_timeout_generation.behavior_logps),
                decoded_text=decoded,
                action_parse_status=parse_status,  # type: ignore[arg-type]
                tool=tool,
            )
        )
    identity = sha256_bytes(
        canonical_json(
            {
                "run_id": run_id,
                "iteration": iteration,
                "task_id": task.card.id,
                "rollout_index": rollout_index,
                "policy_version": policy_version,
                "sampled_ids": [action.sampled_token_ids for action in actions],
            }
        )
    )
    return AgentRlEpisode(
        episode_id=f"agent_rl_{identity[:24]}",
        run_id=run_id,
        iteration=iteration,
        collected_policy_version=policy_version,
        task_id=task.card.id,
        task_version=task.card.version,
        snapshot_sha256=task.card.snapshot_sha256,
        rollout_index=rollout_index,
        actions=actions,
        finish_reason=trajectory.finish_reason,
        hidden_outcome_reward=trajectory.verification.final_score,
        hidden_passed=trajectory.verification.hidden.passed,
        hidden_total=trajectory.verification.hidden.total,
        public_passed=trajectory.verification.public.status == "passed",
        policy_violations=sum(len(step.policy_violations) for step in trajectory.steps),
    )


def generate_agent_rl_group(
    model: Any,
    tokenizer: Any,
    renderer: ChatRenderer,
    task: LoadedAgentTask,
    experiment: AgentRlExperiment,
    backend: SandboxBackend,
    limits: SandboxLimits,
    *,
    run_id: str,
    iteration: int,
    policy_version: int,
    checkpoint_id: str,
) -> AgentRlGroup:
    """Collect independent episodes from the same snapshot and assign group-relative returns."""

    episodes = [
        generate_agent_rl_episode(
            model,
            tokenizer,
            renderer,
            task,
            experiment,
            backend,
            limits,
            run_id=run_id,
            iteration=iteration,
            rollout_index=index,
            policy_version=policy_version,
            checkpoint_id=checkpoint_id,
        )
        for index in range(experiment.rollout.group_size)
    ]
    rewards = torch.tensor(
        [[episode.hidden_outcome_reward for episode in episodes]], dtype=torch.float32
    )
    relative = group_relative_advantages(
        rewards,
        mode=experiment.advantage.mode,
        epsilon=experiment.advantage.epsilon,
    )
    for index, episode in enumerate(episodes):
        advantage = float(relative.advantages[0, index].item())
        episode.advantage = advantage
        for action in episode.actions:
            action.advantage = advantage
    identity = sha256_bytes(
        canonical_json(
            {
                "run_id": run_id,
                "iteration": iteration,
                "task_id": task.card.id,
                "policy_version": policy_version,
                "episodes": [episode.episode_id for episode in episodes],
            }
        )
    )
    return AgentRlGroup(
        group_id=f"agent_rl_group_{identity[:20]}",
        run_id=run_id,
        iteration=iteration,
        policy_version=policy_version,
        task_id=task.card.id,
        snapshot_sha256=task.card.snapshot_sha256,
        reward_mean=float(relative.group_mean[0].item()),
        reward_std=float(relative.group_std[0].item()),
        advantage_mode=experiment.advantage.mode,
        degenerate=bool(relative.degenerate_groups[0].item()),
        episodes=episodes,
    )
