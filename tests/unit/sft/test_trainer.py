from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from nanopt.config.models import OptimizerConfig
from nanopt.models.renderer import RenderedSupervisedExample
from nanopt.sft.data import CompletionOnlyCollator
from nanopt.sft.objective import completion_only_objective
from nanopt.sft.trainer import build_sft_optimizer, train_sft


class TinySftModel(nn.Module):
    """A token-transition table whose only trainable tensor is intentionally LoRA-named."""

    def __init__(self, vocabulary_size: int = 8) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.lora_weight = nn.Parameter(torch.zeros(vocabulary_size, vocabulary_size))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        logits = self.lora_weight[input_ids]
        return SimpleNamespace(logits=logits)


def _example() -> RenderedSupervisedExample:
    return RenderedSupervisedExample(
        input_ids=(1, 2, 3, 4),
        attention_mask=(True, True, True, True),
        action_mask=(False, False, True, True),
        prompt_length=2,
        chat_template_sha256="a" * 64,
    )


def _config(*, max_steps: int = 4) -> OptimizerConfig:
    return OptimizerConfig(
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        epochs=8,
        learning_rate=0.5,
        weight_decay=0.0,
        warmup_ratio=0.0,
        scheduler="cosine",
        max_grad_norm=10.0,
        gradient_checkpointing=False,
        compute_dtype="float32",
        optimizer="adamw",
        log_every_optimizer_steps=1,
        eval_every_optimizer_steps=1,
        save_every_optimizer_steps=1,
    )


def _loss(model: TinySftModel) -> float:
    batch = CompletionOnlyCollator(pad_token_id=0, max_sequence_length=8)([_example()])
    logits = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    ).logits
    return float(completion_only_objective(logits, batch.input_ids, batch.action_mask).loss.item())


def test_repeated_tiny_batch_lowers_completion_loss() -> None:
    model = TinySftModel()
    initial = _loss(model)

    train_sft(
        model,
        [_example()],
        CompletionOnlyCollator(pad_token_id=0, max_sequence_length=8),
        _config(max_steps=6),
        seed=7,
        device=torch.device("cpu"),
    )

    assert _loss(model) < initial * 0.5


def test_resume_at_optimizer_boundary_matches_uninterrupted_training() -> None:
    config = _config(max_steps=4)
    collator = CompletionOnlyCollator(pad_token_id=0, max_sequence_length=8)
    uninterrupted = TinySftModel()
    train_sft(
        uninterrupted,
        [_example()],
        collator,
        config,
        seed=11,
        device=torch.device("cpu"),
    )

    interrupted = TinySftModel()
    first_optimizer = build_sft_optimizer(interrupted, config)
    first = train_sft(
        interrupted,
        [_example()],
        collator,
        config,
        seed=11,
        device=torch.device("cpu"),
        optimizer=first_optimizer,
        stop_after_step=2,
    )
    saved_model = deepcopy(interrupted.state_dict())
    saved_optimizer = deepcopy(first.optimizer.state_dict())

    resumed = TinySftModel()
    resumed.load_state_dict(saved_model)
    resumed_optimizer = build_sft_optimizer(resumed, config)
    resumed_optimizer.load_state_dict(saved_optimizer)
    train_sft(
        resumed,
        [_example()],
        collator,
        config,
        seed=11,
        device=torch.device("cpu"),
        starting_step=first.state.optimizer_step,
        optimizer=resumed_optimizer,
    )

    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
