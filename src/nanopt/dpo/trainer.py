"""Readable LoRA-only Direct Preference Optimization loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from nanopt.config.models import DpoTrainingConfig
from nanopt.core.dpo import DpoLossResult, dpo_loss
from nanopt.core.logprobs import completion_sequence_logps
from nanopt.dpo.data import DpoBatch, PreferenceCollator, RenderedPreferencePair
from nanopt.sft.schedule import cosine_learning_rate, optimizer_groups, select_examples
from nanopt.sft.trainer import trainable_adapter_parameters


@dataclass(frozen=True)
class DpoBatchMetrics:
    """Pair-weighted objective values for one batch or dataset pass."""

    loss: float
    policy_chosen_logp: float
    policy_rejected_logp: float
    policy_margin: float
    reference_margin: float
    implicit_reward_margin: float
    preference_accuracy: float
    reward_accuracy: float
    pair_count: int
    chosen_active_tokens: float
    rejected_active_tokens: float


@dataclass(frozen=True)
class DpoStepMetrics(DpoBatchMetrics):
    optimizer_step: int
    learning_rate: float
    gradient_norm: float
    gradient_clipped: bool
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass(frozen=True)
class DpoTrainingResult:
    optimizer_steps: int
    optimizer: torch.optim.Optimizer
    metrics: list[DpoStepMetrics]


def build_dpo_optimizer(model: torch.nn.Module, config: DpoTrainingConfig) -> torch.optim.AdamW:
    """Build AdamW over the explicitly checked trainable adapter parameters only."""

    parameters = trainable_adapter_parameters(model)
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _pad_width(tensor: Tensor, width: int, *, value: int | bool) -> Tensor:
    if tensor.shape[1] == width:
        return tensor
    padding = torch.full(
        (tensor.shape[0], width - tensor.shape[1]),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=1)


def policy_sequence_logps(
    model: Any,
    batch: DpoBatch,
    *,
    concatenate_chosen_rejected: bool,
) -> tuple[Tensor, Tensor]:
    """Return FP32 chosen/rejected sequence sums under the trainable policy.

    Concatenated mode pads both sides to one width and performs one model forward. Padding token ID
    zero is never attended or scored; it is merely a safe in-vocabulary storage value.
    """

    if not concatenate_chosen_rejected:
        chosen_logits = model(
            input_ids=batch.chosen.input_ids,
            attention_mask=batch.chosen.attention_mask,
            use_cache=False,
        ).logits
        rejected_logits = model(
            input_ids=batch.rejected.input_ids,
            attention_mask=batch.rejected.attention_mask,
            use_cache=False,
        ).logits
        return (
            completion_sequence_logps(
                chosen_logits, batch.chosen.input_ids, batch.chosen.action_mask
            ),
            completion_sequence_logps(
                rejected_logits, batch.rejected.input_ids, batch.rejected.action_mask
            ),
        )

    width = max(batch.chosen.input_ids.shape[1], batch.rejected.input_ids.shape[1])
    chosen_ids = _pad_width(batch.chosen.input_ids, width, value=0)
    rejected_ids = _pad_width(batch.rejected.input_ids, width, value=0)
    chosen_attention = _pad_width(batch.chosen.attention_mask, width, value=False)
    rejected_attention = _pad_width(batch.rejected.attention_mask, width, value=False)
    chosen_actions = _pad_width(batch.chosen.action_mask, width, value=False)
    rejected_actions = _pad_width(batch.rejected.action_mask, width, value=False)
    input_ids = torch.cat((chosen_ids, rejected_ids), dim=0)
    attention_mask = torch.cat((chosen_attention, rejected_attention), dim=0)
    action_mask = torch.cat((chosen_actions, rejected_actions), dim=0)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    sequence_logps = completion_sequence_logps(logits, input_ids, action_mask)
    pair_count = chosen_ids.shape[0]
    return sequence_logps[:pair_count], sequence_logps[pair_count:]


def _objective(model: Any, batch: DpoBatch, config: DpoTrainingConfig) -> DpoLossResult:
    chosen, rejected = policy_sequence_logps(
        model,
        batch,
        concatenate_chosen_rejected=config.concatenate_chosen_rejected,
    )
    return dpo_loss(
        chosen,
        rejected,
        batch.reference_chosen_logps,
        batch.reference_rejected_logps,
        beta=config.beta,
    )


def _batch_metrics(
    result: DpoLossResult,
    chosen_logps: Tensor,
    rejected_logps: Tensor,
    batch: DpoBatch,
) -> DpoBatchMetrics:
    """Detach every diagnostic while keeping the differentiable result private to the caller."""

    chosen_tokens = batch.chosen.action_mask[:, 1:].sum(dim=1).float()
    rejected_tokens = batch.rejected.action_mask[:, 1:].sum(dim=1).float()
    pair_count = len(batch.pair_ids)
    return DpoBatchMetrics(
        loss=float(result.loss.detach().item()),
        policy_chosen_logp=float(chosen_logps.detach().mean().item()),
        policy_rejected_logp=float(rejected_logps.detach().mean().item()),
        policy_margin=float(result.policy_margin.detach().mean().item()),
        reference_margin=float(result.reference_margin.detach().mean().item()),
        implicit_reward_margin=float(result.implicit_reward_margin.detach().mean().item()),
        preference_accuracy=float((result.policy_margin.detach() > 0).float().mean().item()),
        reward_accuracy=float((result.implicit_reward_margin.detach() > 0).float().mean().item()),
        pair_count=pair_count,
        chosen_active_tokens=float(chosen_tokens.mean().item()),
        rejected_active_tokens=float(rejected_tokens.mean().item()),
    )


def evaluate_dpo(
    model: Any,
    examples: Sequence[RenderedPreferencePair],
    collator: PreferenceCollator,
    config: DpoTrainingConfig,
    *,
    device: torch.device,
) -> DpoBatchMetrics:
    """Evaluate pair-weighted DPO diagnostics without retaining computation graphs."""

    if not examples:
        raise ValueError("DPO evaluation examples must not be empty")
    was_training = bool(model.training)
    model.eval()
    totals: dict[str, float] = {}
    total_pairs = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(examples), config.pair_micro_batch_size):
                batch = collator(examples[start : start + config.pair_micro_batch_size]).to(device)
                chosen, rejected = policy_sequence_logps(
                    model,
                    batch,
                    concatenate_chosen_rejected=config.concatenate_chosen_rejected,
                )
                result = dpo_loss(
                    chosen,
                    rejected,
                    batch.reference_chosen_logps,
                    batch.reference_rejected_logps,
                    beta=config.beta,
                )
                metric = _batch_metrics(result, chosen, rejected, batch)
                for name, value in metric.__dict__.items():
                    if name != "pair_count":
                        totals[name] = totals.get(name, 0.0) + float(value) * metric.pair_count
                total_pairs += metric.pair_count
    finally:
        if was_training:
            model.train()
    return DpoBatchMetrics(
        loss=totals["loss"] / total_pairs,
        policy_chosen_logp=totals["policy_chosen_logp"] / total_pairs,
        policy_rejected_logp=totals["policy_rejected_logp"] / total_pairs,
        policy_margin=totals["policy_margin"] / total_pairs,
        reference_margin=totals["reference_margin"] / total_pairs,
        implicit_reward_margin=totals["implicit_reward_margin"] / total_pairs,
        preference_accuracy=totals["preference_accuracy"] / total_pairs,
        reward_accuracy=totals["reward_accuracy"] / total_pairs,
        pair_count=total_pairs,
        chosen_active_tokens=totals["chosen_active_tokens"] / total_pairs,
        rejected_active_tokens=totals["rejected_active_tokens"] / total_pairs,
    )


def _cuda_peaks(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device))


def train_dpo(
    model: Any,
    examples: Sequence[RenderedPreferencePair],
    collator: PreferenceCollator,
    config: DpoTrainingConfig,
    *,
    seed: int,
    device: torch.device,
    on_step: Callable[[DpoStepMetrics], None] | None = None,
) -> DpoTrainingResult:
    """Optimize the mean pair loss with exact pair-weighted gradient accumulation."""

    if not examples:
        raise ValueError("DPO training examples must not be empty")
    groups = optimizer_groups(
        len(examples),
        micro_batch_size=config.pair_micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        epochs=config.epochs,
        seed=seed,
        max_steps=None,
    )
    optimizer = build_dpo_optimizer(model, config)
    parameters = trainable_adapter_parameters(model)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records: list[DpoStepMetrics] = []
    for step_index, group_indices in enumerate(groups):
        learning_rate = cosine_learning_rate(
            step_index,
            total_optimizer_steps=len(groups),
            warmup_ratio=config.warmup_ratio,
            base_learning_rate=config.learning_rate,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        batches = [
            collator(select_examples(examples, indices)).to(device) for indices in group_indices
        ]
        group_pairs = sum(len(batch.pair_ids) for batch in batches)
        accumulated: dict[str, float] = {}
        for batch in batches:
            chosen, rejected = policy_sequence_logps(
                model,
                batch,
                concatenate_chosen_rejected=config.concatenate_chosen_rejected,
            )
            result = dpo_loss(
                chosen,
                rejected,
                batch.reference_chosen_logps,
                batch.reference_rejected_logps,
                beta=config.beta,
            )
            weight = len(batch.pair_ids) / group_pairs
            torch.autograd.backward(result.loss * weight)
            metric = _batch_metrics(result, chosen, rejected, batch)
            for name, value in metric.__dict__.items():
                if name != "pair_count":
                    accumulated[name] = accumulated.get(name, 0.0) + float(value) * weight
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        gradient_norm = float(gradient_norm_tensor.detach().float().item())
        if not torch.isfinite(gradient_norm_tensor):
            raise FloatingPointError("DPO gradient norm is non-finite")
        optimizer.step()
        allocated, reserved = _cuda_peaks(device)
        record = DpoStepMetrics(
            **accumulated,
            pair_count=group_pairs,
            optimizer_step=step_index + 1,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            gradient_clipped=gradient_norm > config.max_grad_norm,
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,
        )
        records.append(record)
        if on_step is not None:
            on_step(record)
    return DpoTrainingResult(optimizer_steps=len(groups), optimizer=optimizer, metrics=records)
