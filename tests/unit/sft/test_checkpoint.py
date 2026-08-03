from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from nanopt.config.models import LoraConfig, OptimizerConfig
from nanopt.models.adapters import attach_lora_adapter, load_lora_adapter
from nanopt.sft.checkpoint import (
    read_sft_checkpoint,
    restore_sft_optimizer,
    save_sft_checkpoint,
)
from nanopt.sft.trainer import SftTrainingState, build_sft_optimizer


def _base() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=8,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        epochs=2,
        learning_rate=0.1,
        weight_decay=0.0,
        warmup_ratio=0.0,
        scheduler="cosine",
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        compute_dtype="float32",
        optimizer="adamw",
        log_every_optimizer_steps=1,
        eval_every_optimizer_steps=1,
        save_every_optimizer_steps=1,
    )


def test_checkpoint_round_trip_validates_adapter_and_optimizer_hashes(tmp_path: Path) -> None:
    torch.manual_seed(13)
    config = LoraConfig(
        method="lora",
        rank=2,
        alpha=4,
        dropout=0.0,
        bias="none",
        target_modules=["q_proj"],
    )
    adapted = attach_lora_adapter(_base(), config, adapter_name="sft")
    optimizer = build_sft_optimizer(adapted, _optimizer_config())
    for parameter in adapted.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    checkpoint = tmp_path / "step-000001"
    metadata = save_sft_checkpoint(
        adapted,
        optimizer,
        SftTrainingState(optimizer_step=1, total_optimizer_steps=2),
        checkpoint,
        adapter_name="sft",
    )
    verified = read_sft_checkpoint(checkpoint)

    torch.manual_seed(13)
    loaded = load_lora_adapter(
        _base(),
        checkpoint / verified.adapter_path,
        adapter_name="sft",
        trainable=True,
    )
    restored_optimizer = build_sft_optimizer(loaded, _optimizer_config())
    restore_sft_optimizer(restored_optimizer, checkpoint, verified)

    assert metadata == verified
    assert restored_optimizer.state_dict()["state"]

    with (checkpoint / metadata.optimizer_path).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="optimizer checkpoint hash"):
        read_sft_checkpoint(checkpoint)
