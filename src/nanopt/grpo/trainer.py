"""Exact-token synchronous GRPO updates with explicit PPO-style clipping."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from nanopt.config.models import GrpoExperiment
from nanopt.core.clipping import clipped_policy_loss
from nanopt.core.kl import sampled_direct_kl, sampled_k3_kl
from nanopt.core.logprobs import causal_token_logps
from nanopt.core.reductions import masked_mean
from nanopt.grpo.records import GrpoCompletionRecord, GrpoTrajectoryRecord
from nanopt.sft.schedule import cosine_learning_rate
from nanopt.sft.trainer import trainable_adapter_parameters


@dataclass(frozen=True)
class GrpoBatch:
    """Padded stored rollout tokens in full and causal-prediction coordinates.

    ``input_ids``, ``attention_mask``, and ``action_mask`` have shape ``[responses, sequence]``.
    ``old_logprobs`` and optional ``reference_logprobs`` have shape
    ``[responses, sequence - 1]`` after the one causal shift. ``advantages`` has shape
    ``[responses]`` and is broadcast only inside the clipped objective.
    """

    input_ids: Tensor
    attention_mask: Tensor
    action_mask: Tensor
    old_logprobs: Tensor
    reference_logprobs: Tensor | None
    advantages: Tensor

    def to(self, device: torch.device | str) -> GrpoBatch:
        return GrpoBatch(
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
class GrpoUpdateMetrics:
    optimizer_steps: int
    policy_loss: float
    kl_loss: float
    total_loss: float
    clip_fraction: float
    ratio_mean: float
    ratio_p95: float
    current_minus_old_logp_mean: float
    sampled_action_surprisal: float
    active_tokens: int
    learning_rate: float
    gradient_norm: float
    gradient_clipped: bool
    training_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def _flatten_completions(
    trajectories: Sequence[GrpoTrajectoryRecord],
) -> list[tuple[GrpoTrajectoryRecord, GrpoCompletionRecord]]:
    return [
        (trajectory, completion)
        for trajectory in trajectories
        for completion in trajectory.completions
    ]


def collate_grpo_completions(
    values: Sequence[tuple[GrpoTrajectoryRecord, GrpoCompletionRecord]],
    *,
    pad_token_id: int,
) -> GrpoBatch:
    """Pad stored IDs/log probabilities directly; decoded text is never an input."""

    if not values:
        raise ValueError("cannot collate an empty GRPO completion batch")
    if pad_token_id < 0:
        raise ValueError("pad token ID must be non-negative")
    maximum = max(
        len(trajectory.prompt.token_ids) + len(completion.token_ids)
        for trajectory, completion in values
    )
    has_reference = [
        completion.reference_logprobs is not None for _trajectory, completion in values
    ]
    if any(has_reference) and not all(has_reference):
        raise ValueError("GRPO minibatch cannot mix present and absent reference log probabilities")
    input_rows: list[list[int]] = []
    attention_rows: list[list[bool]] = []
    action_rows: list[list[bool]] = []
    old_rows: list[list[float]] = []
    reference_rows: list[list[float]] = []
    advantages: list[float] = []
    for trajectory, completion in values:
        prompt_ids = trajectory.prompt.token_ids
        full_ids = [*prompt_ids, *completion.token_ids]
        padding = maximum - len(full_ids)
        input_rows.append([*full_ids, *([pad_token_id] * padding)])
        attention_rows.append([True] * len(full_ids) + [False] * padding)
        action_rows.append(
            [False] * len(prompt_ids)
            + [bool(value) for value in completion.action_mask]
            + [False] * padding
        )
        # Prediction coordinate j scores full token j+1. The first completion token is therefore
        # stored at prompt_length-1, preceded only by prompt-coordinate zeros.
        old_rows.append([0.0] * (len(prompt_ids) - 1) + completion.old_logprobs + [0.0] * padding)
        if completion.reference_logprobs is not None:
            reference_rows.append(
                [0.0] * (len(prompt_ids) - 1) + completion.reference_logprobs + [0.0] * padding
            )
        advantages.append(completion.advantage)
    return GrpoBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        attention_mask=torch.tensor(attention_rows, dtype=torch.bool),
        action_mask=torch.tensor(action_rows, dtype=torch.bool),
        old_logprobs=torch.tensor(old_rows, dtype=torch.float32),
        reference_logprobs=(
            torch.tensor(reference_rows, dtype=torch.float32) if reference_rows else None
        ),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )


def score_stored_actions(model: Any, batch: GrpoBatch) -> Tensor:
    """Score the exact stored token IDs and return ``[responses, sequence - 1]`` FP32 logps."""

    logits = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    ).logits
    return causal_token_logps(logits, batch.input_ids)


def attach_reference_logps(
    model: Any,
    trajectories: Sequence[GrpoTrajectoryRecord],
    *,
    pad_token_id: int,
    device: torch.device,
) -> None:
    """Score stored IDs under the selected frozen adapter and attach exact action values."""

    values = _flatten_completions(trajectories)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for trajectory, completion in values:
                batch = collate_grpo_completions(
                    [(trajectory, completion)], pad_token_id=pad_token_id
                ).to(device)
                token_logps = score_stored_actions(model, batch)
                prediction_mask = batch.action_mask[:, 1:]
                selected = token_logps[prediction_mask]
                if selected.numel() != len(completion.token_ids):
                    raise ValueError("reference action score count differs from stored token IDs")
                completion.reference_logprobs = [float(value) for value in selected.tolist()]
    finally:
        if was_training:
            model.train()


def build_grpo_optimizer(model: torch.nn.Module, experiment: GrpoExperiment) -> torch.optim.AdamW:
    """Build AdamW only over the active GRPO LoRA adapter parameters."""

    return torch.optim.AdamW(
        trainable_adapter_parameters(model),
        lr=experiment.optimization.learning_rate,
        weight_decay=experiment.optimization.weight_decay,
    )


def _minibatches(
    values: Sequence[tuple[GrpoTrajectoryRecord, GrpoCompletionRecord]],
    *,
    size: int,
    seed: int,
) -> list[list[tuple[GrpoTrajectoryRecord, GrpoCompletionRecord]]]:
    indices = list(range(len(values)))
    random.Random(seed).shuffle(indices)
    return [
        [values[index] for index in indices[start : start + size]]
        for start in range(0, len(indices), size)
    ]


def _cuda_peaks(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device))


def update_grpo_policy(
    model: Any,
    trajectories: Sequence[GrpoTrajectoryRecord],
    experiment: GrpoExperiment,
    optimizer: torch.optim.Optimizer,
    *,
    iteration: int,
    pad_token_id: int,
    device: torch.device,
    clock: Any = time.perf_counter,
) -> GrpoUpdateMetrics:
    """Consume stored rollout IDs in one or more explicit clipped optimizer groups."""

    if not trajectories:
        raise ValueError("GRPO update needs at least one trajectory")
    values = _flatten_completions(trajectories)
    if len(values) < 2:
        raise ValueError("GRPO update needs at least two completions")
    if experiment.optimization.kl_beta > 0 and any(
        completion.reference_logprobs is None for _trajectory, completion in values
    ):
        raise ValueError("positive GRPO KL beta requires frozen-reference log probabilities")
    minibatches = _minibatches(
        values,
        size=experiment.optimization.minibatch_completions,
        seed=experiment.seed + iteration,
    )
    optimizer_groups = [
        minibatches[start : start + experiment.optimization.gradient_accumulation_steps]
        for start in range(0, len(minibatches), experiment.optimization.gradient_accumulation_steps)
    ]
    learning_rate = cosine_learning_rate(
        iteration,
        total_optimizer_steps=experiment.optimization.iterations,
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
    ratios: list[Tensor] = []
    log_differences: list[Tensor] = []
    clip_values: list[float] = []
    old_surprisals: list[Tensor] = []
    active_token_total = 0
    maximum_gradient_norm = 0.0
    clipped_any = False
    optimizer_step_count = 0
    for accumulation_group in optimizer_groups:
        optimizer.zero_grad(set_to_none=True)
        batches = [
            collate_grpo_completions(minibatch, pad_token_id=pad_token_id).to(device)
            for minibatch in accumulation_group
        ]
        if experiment.optimization.loss_normalization == "token_mean":
            group_denominator = sum(int(batch.action_mask[:, 1:].sum().item()) for batch in batches)
        else:
            group_denominator = sum(batch.input_ids.shape[0] for batch in batches)
        for batch in batches:
            current_logps = score_stored_actions(model, batch)
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
                raise FloatingPointError("GRPO total loss is non-finite")
            local_weight = (
                int(prediction_mask.sum().item()) / group_denominator
                if experiment.optimization.loss_normalization == "token_mean"
                else batch.input_ids.shape[0] / group_denominator
            )
            torch.autograd.backward(total_loss * local_weight)
            active = prediction_mask.bool()
            active_token_total += int(active.sum().item())
            policy_losses.append(float(clipped.loss.detach().item()))
            kl_losses.append(float(kl_loss.detach().item()))
            total_losses.append(float(total_loss.detach().item()))
            ratios.append(clipped.ratios.detach()[active].cpu())
            log_differences.append((current_logps.detach() - batch.old_logprobs)[active].cpu())
            clip_values.append(float(clipped.clip_fraction.detach().item()))
            old_surprisals.append((-batch.old_logprobs[active]).detach().cpu())
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            parameters, experiment.optimization.max_grad_norm
        )
        if not bool(torch.isfinite(gradient_norm_tensor).item()):
            raise FloatingPointError("GRPO gradient norm is non-finite")
        gradient_norm = float(gradient_norm_tensor.detach().float().item())
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        clipped_any = clipped_any or gradient_norm > experiment.optimization.max_grad_norm
        optimizer.step()
        optimizer_step_count += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ratio_values = torch.cat(ratios).float()
    difference_values = torch.cat(log_differences).float()
    surprisal_values = torch.cat(old_surprisals).float()
    allocated, reserved = _cuda_peaks(device)
    return GrpoUpdateMetrics(
        optimizer_steps=optimizer_step_count,
        policy_loss=sum(policy_losses) / len(policy_losses),
        kl_loss=sum(kl_losses) / len(kl_losses),
        total_loss=sum(total_losses) / len(total_losses),
        clip_fraction=sum(clip_values) / len(clip_values),
        ratio_mean=float(ratio_values.mean().item()),
        ratio_p95=float(torch.quantile(ratio_values, 0.95).item()),
        current_minus_old_logp_mean=float(difference_values.mean().item()),
        sampled_action_surprisal=float(surprisal_values.mean().item()),
        active_tokens=active_token_total,
        learning_rate=learning_rate,
        gradient_norm=maximum_gradient_norm,
        gradient_clipped=clipped_any,
        training_seconds=max(0.0, clock() - started),
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
    )
