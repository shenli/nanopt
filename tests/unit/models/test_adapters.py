"""LoRA lifecycle tests use a randomly initialized tiny local causal model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from peft import get_peft_model_state_dict
from transformers import Qwen3Config, Qwen3ForCausalLM

from nanopt.config.models import LoraConfig
from nanopt.models.adapters import (
    AdapterError,
    attach_lora_adapter,
    clone_lora_adapter,
    freeze_adapter,
    load_lora_adapter,
    parameter_counts,
    save_lora_adapter,
    selected_adapter,
    validate_target_modules,
)


def _tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
    )
    return Qwen3ForCausalLM(config)


def _lora_config() -> LoraConfig:
    return LoraConfig(
        method="lora",
        rank=2,
        alpha=4,
        dropout=0.0,
        bias="none",
        target_modules=["q_proj"],
    )


def test_attach_validates_targets_and_exposes_parameter_counts() -> None:
    model = _tiny_model()
    matches = validate_target_modules(model, ["q_proj"])

    adapted = attach_lora_adapter(model, _lora_config(), adapter_name="sft")
    counts = parameter_counts(adapted)

    assert matches["q_proj"] == ("model.layers.0.self_attn.q_proj",)
    assert counts.total > counts.trainable > 0
    assert all(
        "lora_" in name for name, parameter in adapted.named_parameters() if parameter.requires_grad
    )


def test_clone_freeze_and_temporary_selection_preserve_adapter_state() -> None:
    adapted = attach_lora_adapter(_tiny_model(), _lora_config(), adapter_name="sft")
    for name, parameter in adapted.named_parameters():
        if "lora_" in name and ".sft." in name:
            parameter.data.fill_(0.25)

    clone_lora_adapter(adapted, source_name="sft", target_name="dpo", trainable=True)
    source = get_peft_model_state_dict(
        adapted,
        adapter_name="sft",
        save_embedding_layers=False,
    )
    target = get_peft_model_state_dict(
        adapted,
        adapter_name="dpo",
        save_embedding_layers=False,
    )

    assert source.keys() == target.keys()
    for key in source:
        torch.testing.assert_close(source[key], target[key])
    assert adapted.active_adapter == "dpo"
    assert any(
        parameter.requires_grad for name, parameter in adapted.named_parameters() if ".dpo." in name
    )

    freeze_adapter(adapted, "dpo")
    assert not any(
        parameter.requires_grad for name, parameter in adapted.named_parameters() if ".dpo." in name
    )
    with selected_adapter(adapted, "sft"):
        assert adapted.active_adapter == "sft"
    assert adapted.active_adapter == "dpo"


def test_adapter_save_and_load_preserve_logits(tmp_path: Path) -> None:
    torch.manual_seed(7)
    adapted = attach_lora_adapter(_tiny_model(), _lora_config(), adapter_name="lesson")
    for name, parameter in adapted.named_parameters():
        if "lora_B" in name and ".lesson." in name:
            parameter.data.fill_(0.1)
    adapted.eval()
    input_ids = torch.tensor([[1, 4, 5, 2]])
    expected = adapted(input_ids=input_ids).logits.detach()

    adapter_dir = save_lora_adapter(adapted, tmp_path / "adapter", adapter_name="lesson")

    torch.manual_seed(7)
    loaded = load_lora_adapter(
        _tiny_model(),
        adapter_dir,
        adapter_name="lesson",
        trainable=False,
    )
    loaded.eval()
    actual = loaded(input_ids=input_ids).logits.detach()

    assert (adapter_dir / "adapter_model.safetensors").is_file()
    torch.testing.assert_close(actual, expected)
    assert parameter_counts(loaded).trainable == 0


def test_adapter_helpers_reject_missing_targets_names_and_paths(tmp_path: Path) -> None:
    model = _tiny_model()
    with pytest.raises(AdapterError, match="at least one"):
        validate_target_modules(model, [])
    with pytest.raises(AdapterError, match="do not exist"):
        validate_target_modules(model, ["missing"])
    with pytest.raises(AdapterError, match="must not be empty"):
        attach_lora_adapter(model, _lora_config(), adapter_name="")

    adapted = attach_lora_adapter(_tiny_model(), _lora_config(), adapter_name="sft")
    with pytest.raises(AdapterError, match="does not exist"):
        freeze_adapter(adapted, "missing")
    with pytest.raises(AdapterError, match="must be new"):
        clone_lora_adapter(adapted, source_name="sft", target_name="sft", trainable=True)
    with pytest.raises(AdapterError, match="missing adapter_config"):
        load_lora_adapter(_tiny_model(), tmp_path, adapter_name="x", trainable=False)
