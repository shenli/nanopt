"""A top-to-bottom SFT loop with visible accumulation and optimizer boundaries."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from nanopt.config.models import OptimizerConfig
from nanopt.models.renderer import RenderedSupervisedExample
from nanopt.sft.data import CompletionOnlyCollator
from nanopt.sft.objective import completion_only_objective
from nanopt.sft.schedule import cosine_learning_rate, optimizer_groups, select_examples


@dataclass(frozen=True)
class SftTrainingState:
    """Progress counted only at safe optimizer boundaries."""

    optimizer_step: int
    total_optimizer_steps: int


@dataclass(frozen=True)
class SftStepMetrics:
    """Inspectable measurements emitted after one optimizer update."""

    optimizer_step: int
    completion_nll: float
    completion_token_accuracy: float
    learning_rate: float
    gradient_norm: float
    gradient_clipped: bool
    active_tokens: int
    tokens_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass
class SftTrainingResult:
    """Final state and optimizer needed for an exact checkpoint."""

    state: SftTrainingState
    optimizer: torch.optim.Optimizer
    metrics: list[SftStepMetrics]


def trainable_adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return trainable LoRA tensors and reject accidental base-model optimization."""

    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not named:
        raise ValueError("SFT model has no trainable parameters")
    unexpected = [name for name, _parameter in named if "lora_" not in name]
    if unexpected:
        raise ValueError(f"non-LoRA parameters are trainable: {', '.join(unexpected[:5])}")
    return [parameter for _name, parameter in named]


def build_sft_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.AdamW:
    """Construct AdamW over only the validated adapter tensors."""

    return torch.optim.AdamW(
        trainable_adapter_parameters(model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _cuda_peaks(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def evaluate_completion_nll(
    model: nn.Module,
    examples: Sequence[RenderedSupervisedExample],
    collator: CompletionOnlyCollator,
    *,
    micro_batch_size: int,
    device: torch.device,
) -> tuple[float, float, int]:
    """Measure token-weighted completion NLL and accuracy without changing parameters."""

    if not examples:
        raise ValueError("validation examples must not be empty")
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_correct = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), micro_batch_size):
            batch = collator(examples[start : start + micro_batch_size]).to(device)
            logits = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            ).logits
            objective = completion_only_objective(logits, batch.input_ids, batch.action_mask)
            total_nll += float(objective.loss.item()) * objective.active_tokens
            total_correct += float(objective.token_accuracy.item()) * objective.active_tokens
            total_tokens += objective.active_tokens
    model.train(was_training)
    return total_nll / total_tokens, total_correct / total_tokens, total_tokens


def train_sft(
    model: nn.Module,
    examples: Sequence[RenderedSupervisedExample],
    collator: CompletionOnlyCollator,
    config: OptimizerConfig,
    *,
    seed: int,
    device: torch.device,
    starting_step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    on_step: Callable[[SftStepMetrics, SftTrainingState, torch.optim.Optimizer], None]
    | None = None,
    stop_after_step: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SftTrainingResult:
    """Train LoRA parameters with token-correct gradient accumulation.

    The schedule is materialized before training. Each outer group is one clean optimizer boundary.
    Micro-batch losses are weighted by active completion tokens, so accumulation is exactly the
    same objective as concatenating the group's unpadded completion targets.
    """

    if not examples:
        raise ValueError("training examples must not be empty")
    groups = optimizer_groups(
        len(examples),
        micro_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        epochs=config.epochs,
        seed=seed,
        max_steps=config.max_steps,
    )
    total_steps = len(groups)
    if starting_step < 0 or starting_step > total_steps:
        raise ValueError("starting_step is outside the deterministic SFT schedule")
    ending_step = total_steps if stop_after_step is None else stop_after_step
    if ending_step < starting_step or ending_step > total_steps:
        raise ValueError("stop_after_step must be between starting_step and the schedule end")
    selected_optimizer = optimizer or build_sft_optimizer(model, config)
    parameters = trainable_adapter_parameters(model)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records: list[SftStepMetrics] = []

    for step_index in range(starting_step, ending_step):
        learning_rate = cosine_learning_rate(
            step_index,
            total_optimizer_steps=total_steps,
            warmup_ratio=config.warmup_ratio,
            base_learning_rate=config.learning_rate,
        )
        for group in selected_optimizer.param_groups:
            group["lr"] = learning_rate
        selected_optimizer.zero_grad(set_to_none=True)

        cpu_batches = [
            collator(select_examples(examples, indices)) for indices in groups[step_index]
        ]
        group_tokens = sum(int(batch.action_mask[:, 1:].sum().item()) for batch in cpu_batches)
        if group_tokens == 0:
            raise ValueError("optimizer group contains no active completion tokens")
        weighted_nll = 0.0
        weighted_accuracy = 0.0
        started = clock()
        for cpu_batch in cpu_batches:
            batch = cpu_batch.to(device)
            logits = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            ).logits
            objective = completion_only_objective(logits, batch.input_ids, batch.action_mask)
            weight = objective.active_tokens / group_tokens
            torch.autograd.backward(objective.loss * weight)
            weighted_nll += float(objective.loss.detach().item()) * weight
            weighted_accuracy += float(objective.token_accuracy.item()) * weight

        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        gradient_norm = float(gradient_norm_tensor.detach().float().item())
        selected_optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = max(clock() - started, 1e-12)
        allocated, reserved = _cuda_peaks(device)
        record = SftStepMetrics(
            optimizer_step=step_index + 1,
            completion_nll=weighted_nll,
            completion_token_accuracy=weighted_accuracy,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            gradient_clipped=gradient_norm > config.max_grad_norm,
            active_tokens=group_tokens,
            tokens_per_second=group_tokens / elapsed,
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,
        )
        state = SftTrainingState(step_index + 1, total_steps)
        records.append(record)
        if on_step is not None:
            on_step(record, state, selected_optimizer)

    return SftTrainingResult(
        state=SftTrainingState(ending_step, total_steps),
        optimizer=selected_optimizer,
        metrics=records,
    )
