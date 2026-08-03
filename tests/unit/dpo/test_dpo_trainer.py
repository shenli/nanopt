from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from nanopt.config.models import DpoTrainingConfig
from nanopt.data.preferences import PreferencePair
from nanopt.dpo.data import (
    CachedReferenceValues,
    PreferenceCollator,
    RenderedPreferencePair,
)
from nanopt.dpo.trainer import evaluate_dpo, train_dpo
from nanopt.models.renderer import RenderedSupervisedExample


class TinyDpoModel(nn.Module):
    """A transition table with a LoRA-named trainable tensor for direction tests."""

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


def _example() -> RenderedPreferencePair:
    pair = PreferencePair(
        pair_id="pair",
        task_id="task",
        family="addition_subtraction",
        difficulty=1,
        split="train",
        prompt="x",
        chosen="y",
        rejected="z",
        rejection_type="wrong_answer",
        source_dataset_fingerprint="source",
        seed=1,
    )

    def sequence(last: int) -> RenderedSupervisedExample:
        return RenderedSupervisedExample(
            input_ids=(1, 2, last),
            attention_mask=(True, True, True),
            action_mask=(False, False, True),
            prompt_length=2,
            chat_template_sha256="a" * 64,
        )

    return RenderedPreferencePair(pair=pair, chosen=sequence(3), rejected=sequence(4))


def _config() -> DpoTrainingConfig:
    return DpoTrainingConfig(
        pair_micro_batch_size=1,
        gradient_accumulation_steps=1,
        epochs=8,
        learning_rate=0.2,
        beta=1.0,
        weight_decay=0.0,
        warmup_ratio=0.0,
        scheduler="cosine",
        max_grad_norm=10.0,
        gradient_checkpointing=False,
        compute_dtype="float32",
        optimizer="adamw",
        concatenate_chosen_rejected=True,
        log_every_optimizer_steps=1,
        eval_every_optimizer_steps=1,
        save_every_optimizer_steps=1,
    )


def test_tiny_dpo_update_increases_chosen_margin() -> None:
    model = TinyDpoModel()
    examples = [_example()]
    collator = PreferenceCollator(
        pad_token_id=0,
        max_prompt_length=8,
        max_completion_length=8,
        reference_values={
            "pair": CachedReferenceValues(
                chosen_logp=-2.0794415,
                rejected_logp=-2.0794415,
                chosen_active_tokens=1,
                rejected_active_tokens=1,
            )
        },
    )
    initial = evaluate_dpo(model, examples, collator, _config(), device=torch.device("cpu"))

    train_dpo(
        model,
        examples,
        collator,
        _config(),
        seed=3,
        device=torch.device("cpu"),
    )
    final = evaluate_dpo(model, examples, collator, _config(), device=torch.device("cpu"))

    assert initial.policy_margin == 0
    assert final.policy_margin > 1
    assert final.loss < initial.loss
