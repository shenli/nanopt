"""White-box clipped policy updates over exact stateful-agent action turns."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from nanopt.agent.rl_records import (
    AgentRlAction,
    AgentRlCreditStudy,
    AgentRlEpisode,
    AgentRlGroup,
    AgentRlStalenessPoint,
)
from nanopt.config.models import AgentRlExperiment
from nanopt.core.clipping import clipped_policy_loss
from nanopt.core.kl import sampled_direct_kl, sampled_k3_kl
from nanopt.core.logprobs import causal_token_logps
from nanopt.core.reductions import masked_mean
from nanopt.sft.schedule import cosine_learning_rate
from nanopt.sft.trainer import trainable_adapter_parameters


@dataclass(frozen=True)
class AgentRlBatch:
    """Padded action turns in full-token and causal-prediction coordinates.

    Full tensors have shape ``[actions, sequence]``. Log probabilities have shape
    ``[actions, sequence - 1]`` because causal prediction coordinate ``j`` scores token ``j+1``.
    ``advantages`` has shape ``[actions]`` and is broadcast across sampled action tokens only.
    """

    input_ids: Tensor
    attention_mask: Tensor
    action_mask: Tensor
    old_logprobs: Tensor
    reference_logprobs: Tensor | None
    advantages: Tensor

    def to(self, device: torch.device | str) -> AgentRlBatch:
        return AgentRlBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
            old_logprobs=self.old_logprobs.to(device),
            reference_logprobs=(
                self.reference_logprobs.to(device) if self.reference_logprobs is not None else None
            ),
            advantages=self.advantages.to(device),
        )


@dataclass(frozen=True)
class AgentRlUpdateMetrics:
    optimizer_steps: int
    policy_loss: float
    kl_loss: float
    total_loss: float
    clip_fraction: float
    ratio_mean: float
    gradient_norm: float
    active_tokens: int
    training_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def flatten_agent_rl_actions(groups: Sequence[AgentRlGroup]) -> list[AgentRlAction]:
    """Return action turns in stable group/episode/turn order."""

    return [action for group in groups for episode in group.episodes for action in episode.actions]


def collate_agent_rl_actions(
    actions: Sequence[AgentRlAction], *, pad_token_id: int
) -> AgentRlBatch:
    """Pad stored prompt/action IDs directly without decoding or re-tokenizing."""

    if not actions:
        raise ValueError("cannot collate an empty Agent RL action batch")
    if pad_token_id < 0:
        raise ValueError("pad token ID must be non-negative")
    maximum = max(
        len(action.prompt_token_ids) + len(action.sampled_token_ids) for action in actions
    )
    has_reference = [action.reference_logprobs is not None for action in actions]
    if any(has_reference) and not all(has_reference):
        raise ValueError("Agent RL batch cannot mix present and absent reference log probabilities")

    input_rows: list[list[int]] = []
    attention_rows: list[list[bool]] = []
    action_rows: list[list[bool]] = []
    old_rows: list[list[float]] = []
    reference_rows: list[list[float]] = []
    advantages: list[float] = []
    for action in actions:
        prompt_length = len(action.prompt_token_ids)
        full_ids = [*action.prompt_token_ids, *action.sampled_token_ids]
        padding = maximum - len(full_ids)
        input_rows.append([*full_ids, *([pad_token_id] * padding)])
        attention_rows.append([True] * len(full_ids) + [False] * padding)
        action_rows.append([False] * prompt_length + list(action.action_mask) + [False] * padding)
        # Prediction coordinate prompt_length-1 scores the first sampled action token.
        old_rows.append([0.0] * (prompt_length - 1) + action.old_logprobs + [0.0] * padding)
        if action.reference_logprobs is not None:
            reference_rows.append(
                [0.0] * (prompt_length - 1) + action.reference_logprobs + [0.0] * padding
            )
        advantages.append(action.advantage)
    return AgentRlBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        attention_mask=torch.tensor(attention_rows, dtype=torch.bool),
        action_mask=torch.tensor(action_rows, dtype=torch.bool),
        old_logprobs=torch.tensor(old_rows, dtype=torch.float32),
        reference_logprobs=(
            torch.tensor(reference_rows, dtype=torch.float32) if reference_rows else None
        ),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )


def score_agent_rl_actions(model: Any, batch: AgentRlBatch) -> Tensor:
    """Return FP32 log probabilities ``[actions, sequence - 1]`` for stored sampled IDs."""

    logits = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    ).logits
    return causal_token_logps(logits, batch.input_ids)


def attach_agent_rl_reference_logps(
    model: Any,
    groups: Sequence[AgentRlGroup],
    *,
    pad_token_id: int,
    device: torch.device,
) -> None:
    """Score exact action IDs under the frozen Agent SFT adapter and attach aligned values."""

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for action in flatten_agent_rl_actions(groups):
                batch = collate_agent_rl_actions([action], pad_token_id=pad_token_id).to(device)
                token_logps = score_agent_rl_actions(model, batch)
                selected = token_logps[batch.action_mask[:, 1:]]
                if selected.numel() != len(action.sampled_token_ids):
                    raise ValueError("reference score count differs from Agent RL sampled IDs")
                action.reference_logprobs = [float(value) for value in selected.tolist()]
    finally:
        if was_training:
            model.train()


def build_agent_rl_optimizer(
    model: torch.nn.Module, experiment: AgentRlExperiment
) -> torch.optim.AdamW:
    """Build AdamW only over the cloned trainable Agent RL adapter."""

    return torch.optim.AdamW(
        trainable_adapter_parameters(model),
        lr=experiment.optimization.learning_rate,
        weight_decay=experiment.optimization.weight_decay,
    )


def _minibatches(
    actions: Sequence[AgentRlAction], *, size: int, seed: int
) -> list[list[AgentRlAction]]:
    indices = list(range(len(actions)))
    random.Random(seed).shuffle(indices)
    return [
        [actions[index] for index in indices[start : start + size]]
        for start in range(0, len(indices), size)
    ]


def update_agent_rl_policy(
    model: Any,
    groups: Sequence[AgentRlGroup],
    experiment: AgentRlExperiment,
    optimizer: torch.optim.Optimizer,
    *,
    iteration: int,
    policy_version: int,
    pad_token_id: int,
    device: torch.device,
    clock: Any = time.perf_counter,
) -> AgentRlUpdateMetrics:
    """Apply one fresh-data clipped update; stale collection versions are rejected."""

    if not groups:
        raise ValueError("Agent RL update needs at least one rollout group")
    if any(group.policy_version != policy_version for group in groups):
        raise ValueError("Agent RL refuses stale rollout groups for policy updates")
    actions = flatten_agent_rl_actions(groups)
    if not actions:
        raise ValueError("Agent RL update needs at least one sampled action")
    if experiment.optimization.kl_beta > 0 and any(
        action.reference_logprobs is None for action in actions
    ):
        raise ValueError("positive Agent RL KL beta requires reference log probabilities")

    minibatches = _minibatches(
        actions,
        size=experiment.optimization.minibatch_actions,
        seed=experiment.seed + iteration,
    )
    accumulation = experiment.optimization.gradient_accumulation_steps
    optimizer_groups = [
        minibatches[start : start + accumulation]
        for start in range(0, len(minibatches), accumulation)
    ]
    learning_rate = cosine_learning_rate(
        iteration,
        total_optimizer_steps=experiment.rollout.iterations,
        warmup_ratio=experiment.optimization.warmup_ratio,
        base_learning_rate=experiment.optimization.learning_rate,
    )
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate

    parameters = trainable_adapter_parameters(model)
    model.train()
    started = clock()
    policy_losses: list[float] = []
    kl_losses: list[float] = []
    total_losses: list[float] = []
    ratio_values: list[Tensor] = []
    clip_values: list[float] = []
    active_token_total = 0
    maximum_gradient_norm = 0.0
    optimizer_steps = 0
    for accumulation_group in optimizer_groups:
        optimizer.zero_grad(set_to_none=True)
        batches = [
            collate_agent_rl_actions(items, pad_token_id=pad_token_id).to(device)
            for items in accumulation_group
        ]
        if experiment.optimization.loss_normalization == "token_mean":
            denominator = sum(int(batch.action_mask[:, 1:].sum().item()) for batch in batches)
        else:
            denominator = sum(batch.input_ids.shape[0] for batch in batches)
        for batch in batches:
            current_logps = score_agent_rl_actions(model, batch)
            prediction_mask = batch.action_mask[:, 1:]
            clipped = clipped_policy_loss(
                current_logps,
                batch.old_logprobs,
                batch.advantages,
                prediction_mask,
                clip_epsilon=experiment.optimization.clip_epsilon,
                normalization=experiment.optimization.loss_normalization,
            )
            if batch.reference_logprobs is None:
                kl_loss = current_logps.new_zeros(())
            else:
                estimator = (
                    sampled_k3_kl(current_logps, batch.reference_logprobs)
                    if experiment.optimization.kl_estimator == "k3"
                    else sampled_direct_kl(current_logps, batch.reference_logprobs)
                )
                kl_loss = masked_mean(estimator, prediction_mask, dim=(0, 1))
            total_loss = clipped.loss + experiment.optimization.kl_beta * kl_loss
            if not bool(torch.isfinite(total_loss).item()):
                raise FloatingPointError("Agent RL total loss is non-finite")
            local_weight = (
                int(prediction_mask.sum().item()) / denominator
                if experiment.optimization.loss_normalization == "token_mean"
                else batch.input_ids.shape[0] / denominator
            )
            torch.autograd.backward(total_loss * local_weight)
            active = prediction_mask.bool()
            active_token_total += int(active.sum().item())
            policy_losses.append(float(clipped.loss.detach().item()))
            kl_losses.append(float(kl_loss.detach().item()))
            total_losses.append(float(total_loss.detach().item()))
            ratio_values.append(clipped.ratios.detach()[active].cpu())
            clip_values.append(float(clipped.clip_fraction.detach().item()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, experiment.optimization.max_grad_norm
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError("Agent RL gradient norm is non-finite")
        maximum_gradient_norm = max(
            maximum_gradient_norm, float(gradient_norm.detach().float().item())
        )
        optimizer.step()
        optimizer_steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    ratios = torch.cat(ratio_values).float()
    return AgentRlUpdateMetrics(
        optimizer_steps=optimizer_steps,
        policy_loss=sum(policy_losses) / len(policy_losses),
        kl_loss=sum(kl_losses) / len(kl_losses),
        total_loss=sum(total_losses) / len(total_losses),
        clip_fraction=sum(clip_values) / len(clip_values),
        ratio_mean=float(ratios.mean().item()),
        gradient_norm=maximum_gradient_norm,
        active_tokens=active_token_total,
        training_seconds=max(0.0, clock() - started),
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
    )


def measure_agent_rl_staleness(
    model: Any,
    groups: Sequence[AgentRlGroup],
    *,
    label: str,
    scored_policy_version: int,
    pad_token_id: int,
    device: torch.device,
) -> AgentRlStalenessPoint:
    """Measure importance-ratio drift without allowing retained data into an update."""

    actions = flatten_agent_rl_actions(groups)
    if not actions:
        raise ValueError("staleness study needs at least one action")
    collected_versions = {group.policy_version for group in groups}
    if len(collected_versions) != 1:
        raise ValueError("one staleness point must use a single collection policy version")
    collected = next(iter(collected_versions))
    if scored_policy_version < collected:
        raise ValueError("scored policy version cannot precede collection")
    model.eval()
    differences: list[Tensor] = []
    with torch.inference_mode():
        for action in actions:
            batch = collate_agent_rl_actions([action], pad_token_id=pad_token_id).to(device)
            current = score_agent_rl_actions(model, batch)
            mask = batch.action_mask[:, 1:]
            differences.append((current - batch.old_logprobs)[mask].detach().float().cpu())
    log_ratios = torch.cat(differences)
    ratios = log_ratios.exp()
    ess_fraction = (ratios.sum().square() / (ratios.numel() * ratios.square().sum())).item()
    return AgentRlStalenessPoint(
        label=label,  # type: ignore[arg-type]
        collected_policy_version=collected,
        scored_policy_version=scored_policy_version,
        policy_lag=scored_policy_version - collected,
        active_tokens=log_ratios.numel(),
        mean_abs_log_ratio=float(log_ratios.abs().mean().item()),
        max_abs_log_ratio=float(log_ratios.abs().max().item()),
        approximate_ess_fraction=float(ess_fraction),
    )


def build_credit_assignment_study(groups: Sequence[AgentRlGroup]) -> AgentRlCreditStudy:
    """Compare token coverage for all-action and terminal-action credit assignments."""

    episodes: list[AgentRlEpisode] = [episode for group in groups for episode in group.episodes]
    if not episodes:
        raise ValueError("credit-assignment study needs at least one episode")
    all_tokens = 0
    terminal_tokens = 0
    by_tool: dict[str, int] = {}
    for episode in episodes:
        for action in episode.actions:
            tokens = sum(action.action_mask)
            all_tokens += tokens
            key = action.tool or "invalid_action"
            by_tool[key] = by_tool.get(key, 0) + tokens
        terminal_tokens += sum(episode.actions[-1].action_mask)
    return AgentRlCreditStudy(
        episodes=len(episodes),
        successful_episodes=sum(episode.hidden_outcome_reward == 1.0 for episode in episodes),
        all_actions_active_tokens=all_tokens,
        terminal_action_active_tokens=terminal_tokens,
        terminal_token_fraction=terminal_tokens / all_tokens,
        active_tokens_by_tool=dict(sorted(by_tool.items())),
    )
