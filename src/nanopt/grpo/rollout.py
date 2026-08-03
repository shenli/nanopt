"""Fresh grouped autoregressive rollouts with exact sampled-token evidence."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from nanopt.config.models import GrpoExperiment
from nanopt.core.advantages import group_relative_advantages
from nanopt.data.schemas import ArithmeticTask
from nanopt.grpo.records import (
    GrpoCompletionRecord,
    GrpoPromptRecord,
    GrpoTrajectoryRecord,
    TrajectoryFinishReason,
)
from nanopt.grpo.reward import arithmetic_rlvr_reward
from nanopt.models.renderer import ChatRenderer
from nanopt.rollout.sampler import SamplingConfig, sample_autoregressive
from nanopt.runtime.artifacts import canonical_json, sha256_bytes


def rollout_seed(base_seed: int, iteration: int, task_id: str, completion_index: int) -> int:
    """Derive one stable 63-bit seed without consuming global RNG state."""

    if iteration < 0 or completion_index < 0 or not task_id:
        raise ValueError("rollout seed coordinates are invalid")
    value = f"grpo-rollout-v1\0{base_seed}\0{iteration}\0{task_id}\0{completion_index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


def _finish_reason(value: str) -> TrajectoryFinishReason:
    if value == "eos":
        return "eos"
    if value == "stop_sequence":
        return "protocol_stop"
    if value == "length":
        return "max_length"
    return "error"


def generate_grouped_trajectory(
    model: Any,
    tokenizer: Any,
    renderer: ChatRenderer,
    task: ArithmeticTask,
    experiment: GrpoExperiment,
    *,
    run_id: str,
    iteration: int,
    eos_token_id: int,
    stop_token_sequence: tuple[int, ...],
    clock: Callable[[], float] = time.perf_counter,
) -> GrpoTrajectoryRecord:
    """Sample, decode for reward only, standardize rewards, and freeze one prompt group."""

    if experiment.rollout.top_k is not None:
        raise ValueError("the exact v0.1 training sampler requires top_k=null")
    prompt_messages = [{"role": "user", "content": task.prompt}]
    prompt = renderer.render_prompt(prompt_messages)
    sampling = SamplingConfig(
        max_new_tokens=experiment.rollout.max_completion_length,
        do_sample=True,
        temperature=experiment.rollout.temperature,
        top_p=experiment.rollout.top_p,
        eos_token_id=eos_token_id if experiment.rollout.stop_on_eos else None,
        stop_token_sequences=(stop_token_sequence,),
    )
    provisional: list[tuple[Any, str, Any, float]] = []
    for completion_index in range(experiment.rollout.group_size):
        started = clock()
        generation = sample_autoregressive(
            model,
            torch.tensor(prompt.input_ids, dtype=torch.long),
            sampling,
            seed=rollout_seed(experiment.seed, iteration, task.task_id, completion_index),
        )
        elapsed = max(0.0, clock() - started)
        response = tokenizer.decode(list(generation.generated_token_ids), skip_special_tokens=True)
        if not isinstance(response, str):
            raise TypeError("tokenizer decode must return text")
        reward = arithmetic_rlvr_reward(
            task,
            response,
            experiment.reward.components,
            completion_tokens=len(generation.generated_token_ids),
        )
        provisional.append((generation, response, reward, elapsed))
    reward_tensor = torch.tensor(
        [[float(item[2].reward) for item in provisional]], dtype=torch.float32
    )
    advantages = group_relative_advantages(
        reward_tensor,
        mode=experiment.advantage.mode,
        epsilon=experiment.advantage.epsilon,
    )
    completions: list[GrpoCompletionRecord] = []
    for completion_index, (generation, response, reward, elapsed) in enumerate(provisional):
        # At temperature=1 and top_p=1 the behavior distribution exactly equals the policy
        # distribution. Storing behavior_logps makes the on-policy contract explicit.
        completions.append(
            GrpoCompletionRecord(
                completion_index=completion_index,
                token_ids=list(generation.generated_token_ids),
                action_mask=[int(value) for value in generation.active_mask],
                old_logprobs=list(generation.behavior_logps),
                decoded_text=response,
                finish_reason=_finish_reason(generation.finish_reason),
                reward=reward.reward,
                reward_components=reward.components,
                advantage=float(advantages.advantages[0, completion_index].item()),
                parser_status=reward.parser_status,
                parsed_answer=reward.parsed_answer,
                verifier_status=reward.verifier_status,
                generation_seconds=elapsed,
            )
        )
    identity = sha256_bytes(
        canonical_json(
            {
                "run_id": run_id,
                "iteration": iteration,
                "task_id": task.task_id,
                "completion_ids": [item.token_ids for item in completions],
            }
        )
    )
    return GrpoTrajectoryRecord(
        trajectory_id=f"rlvr_{identity[:24]}",
        run_id=run_id,
        iteration=iteration,
        task_id=task.task_id,
        prompt=GrpoPromptRecord(
            messages=prompt_messages,
            token_ids=list(prompt.input_ids),
            attention_mask=[1] * len(prompt.input_ids),
        ),
        group_reward_mean=float(advantages.group_mean[0].item()),
        group_reward_std=float(advantages.group_std[0].item()),
        advantage_mode=experiment.advantage.mode,
        completions=completions,
    )


def deterministic_prompt_schedule(
    tasks: Sequence[ArithmeticTask], *, iterations: int, batch_size: int, seed: int
) -> list[list[ArithmeticTask]]:
    """Materialize repeatable shuffled prompt batches for independently inspectable iterations."""

    if not tasks or iterations <= 0 or batch_size <= 0:
        raise ValueError("prompt schedule inputs must be non-empty and positive")
    import random

    rng = random.Random(seed)
    pool = sorted(tasks, key=lambda task: task.task_id)
    order: list[ArithmeticTask] = []
    needed = iterations * batch_size
    while len(order) < needed:
        epoch = list(pool)
        rng.shuffle(epoch)
        order.extend(epoch)
    return [order[start : start + batch_size] for start in range(0, needed, batch_size)]
