"""CPU lab: watch a completion-only objective train a tiny token-transition model."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from nanopt.config.models import OptimizerConfig
from nanopt.models.renderer import RenderedSupervisedExample
from nanopt.sft.data import CompletionOnlyCollator
from nanopt.sft.objective import completion_only_objective
from nanopt.sft.trainer import train_sft


class TinyPolicy(nn.Module):
    """Learn next-token transitions with one deliberately LoRA-named matrix."""

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


def main() -> None:
    example = RenderedSupervisedExample(
        input_ids=(1, 2, 3, 4),
        attention_mask=(True, True, True, True),
        action_mask=(False, False, True, True),
        prompt_length=2,
        chat_template_sha256="lab",
    )
    collator = CompletionOnlyCollator(pad_token_id=0, max_sequence_length=8)
    batch = collator([example])
    model = TinyPolicy()

    def loss() -> float:
        logits = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        ).logits
        return float(completion_only_objective(logits, batch.input_ids, batch.action_mask).loss)

    initial = loss()
    config = OptimizerConfig(
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=6,
        epochs=6,
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
    result = train_sft(
        model,
        [example],
        collator,
        config,
        seed=42,
        device=torch.device("cpu"),
    )
    final = loss()

    assert batch.action_mask.tolist() == [[False, False, True, True]]
    assert result.state.optimizer_step == 6
    assert final < initial
    print(f"Action mask: {batch.action_mask.tolist()[0]}")
    print(f"Initial completion NLL: {initial:.4f}")
    print(f"Final completion NLL:   {final:.4f}")
    print("Completion-only SFT lab passed.")


if __name__ == "__main__":
    main()
