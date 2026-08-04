"""Explicit LoRA adapter creation, cloning, selection, persistence, and counts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import nn
from transformers import PreTrainedModel

from nanopt.config.models import LoraConfig as NanoLoraConfig


class AdapterError(ValueError):
    """Raised when a requested adapter lifecycle operation is ambiguous or unsafe."""


@dataclass(frozen=True)
class ParameterCounts:
    """Total and trainable scalar parameter counts."""

    total: int
    trainable: int


def parameter_counts(model: nn.Module) -> ParameterCounts:
    """Count all and ``requires_grad`` parameters without relying on model-specific helpers."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ParameterCounts(total=total, trainable=trainable)


def validate_target_modules(
    model: nn.Module, target_modules: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Map every requested LoRA target suffix to concrete module names.

    A target ``q_proj`` matches modules named exactly ``q_proj`` or ending in ``.q_proj``. Every
    requested target must match at least once; otherwise a model architecture change could silently
    produce a no-op adapter.
    """

    if not target_modules or any(not target for target in target_modules):
        raise AdapterError("target_modules must contain at least one non-empty name")
    module_names = tuple(name for name, _module in model.named_modules())
    matches = {
        target: tuple(
            name for name in module_names if name == target or name.endswith(f".{target}")
        )
        for target in target_modules
    }
    missing = [target for target, names in matches.items() if not names]
    if missing:
        raise AdapterError(f"LoRA target modules do not exist: {', '.join(sorted(missing))}")
    return matches


def _peft_config(config: NanoLoraConfig) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=config.bias,
        target_modules=list(config.target_modules),
    )


def attach_lora_adapter(
    model: PreTrainedModel,
    config: NanoLoraConfig,
    *,
    adapter_name: str,
) -> PeftModel:
    """Validate target modules and attach one named, trainable causal-LM LoRA adapter."""

    if not adapter_name:
        raise AdapterError("adapter_name must not be empty")
    validate_target_modules(model, config.target_modules)
    return cast(PeftModel, get_peft_model(model, _peft_config(config), adapter_name=adapter_name))


def _require_adapter(model: PeftModel, adapter_name: str) -> None:
    if adapter_name not in model.peft_config:
        raise AdapterError(f"adapter {adapter_name!r} does not exist")


def _set_only_adapter_trainable(model: PeftModel, adapter_name: str, *, trainable: bool) -> None:
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable and "lora_" in name and marker in name


def freeze_adapter(model: PeftModel, adapter_name: str) -> None:
    """Freeze every parameter belonging to one named adapter."""

    _require_adapter(model, adapter_name)
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        if "lora_" in name and marker in name:
            parameter.requires_grad = False


def snapshot_lora_adapter(model: PeftModel, adapter_name: str) -> dict[str, torch.Tensor]:
    """Copy one adapter to CPU tensors for explicit in-run checkpoint selection."""

    _require_adapter(model, adapter_name)
    state = get_peft_model_state_dict(
        model,
        adapter_name=adapter_name,
        save_embedding_layers=False,
    )
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def restore_lora_adapter(
    model: PeftModel,
    adapter_name: str,
    state: dict[str, torch.Tensor],
) -> None:
    """Restore a trusted in-memory adapter snapshot without changing its trainability."""

    _require_adapter(model, adapter_name)
    expected = get_peft_model_state_dict(
        model,
        adapter_name=adapter_name,
        save_embedding_layers=False,
    )
    if state.keys() != expected.keys():
        raise AdapterError("adapter snapshot keys do not match the selected adapter")
    set_peft_model_state_dict(model, state, adapter_name=adapter_name)


def clone_lora_adapter(
    model: PeftModel,
    *,
    source_name: str,
    target_name: str,
    trainable: bool,
) -> None:
    """Copy adapter configuration and weights into a new named adapter."""

    _require_adapter(model, source_name)
    if not target_name or target_name in model.peft_config:
        raise AdapterError(f"target adapter name must be new and non-empty: {target_name!r}")
    source_config = deepcopy(model.peft_config[source_name])
    model.add_adapter(target_name, source_config)
    source_state = get_peft_model_state_dict(
        model,
        adapter_name=source_name,
        save_embedding_layers=False,
    )
    set_peft_model_state_dict(model, source_state, adapter_name=target_name)
    target_state = get_peft_model_state_dict(
        model,
        adapter_name=target_name,
        save_embedding_layers=False,
    )
    if source_state.keys() != target_state.keys() or any(
        not torch.equal(source_state[key], target_state[key]) for key in source_state
    ):
        raise AdapterError("cloned adapter weights do not match the source adapter")
    model.set_adapter(target_name, inference_mode=not trainable)
    _set_only_adapter_trainable(model, target_name, trainable=trainable)


@contextmanager
def selected_adapter(model: PeftModel, adapter_name: str) -> Iterator[None]:
    """Select one adapter temporarily and restore the previously active adapter afterward."""

    _require_adapter(model, adapter_name)
    previous = model.active_adapter
    if not isinstance(previous, str):
        raise AdapterError("selected_adapter supports one previously active adapter")
    model.set_adapter(adapter_name)
    try:
        yield
    finally:
        model.set_adapter(previous)


def save_lora_adapter(model: PeftModel, path: Path, *, adapter_name: str) -> Path:
    """Save one named adapter with safetensors and return its concrete directory."""

    _require_adapter(model, adapter_name)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        str(path),
        selected_adapters=[adapter_name],
        safe_serialization=True,
        save_embedding_layers=False,
    )
    # PEFT stores a non-default named adapter in a subdirectory of the requested destination.
    candidate = path / adapter_name
    adapter_dir = candidate if candidate.is_dir() else path
    if not (adapter_dir / "adapter_config.json").is_file():
        raise AdapterError(f"PEFT did not write adapter_config.json under {adapter_dir}")
    return adapter_dir


def load_lora_adapter(
    model: PreTrainedModel,
    path: Path,
    *,
    adapter_name: str,
    trainable: bool,
) -> PeftModel:
    """Load one saved adapter onto a base model without downloading any files."""

    if not (path / "adapter_config.json").is_file():
        raise AdapterError(f"adapter directory is missing adapter_config.json: {path}")
    loaded = PeftModel.from_pretrained(
        model,
        str(path),
        adapter_name=adapter_name,
        is_trainable=trainable,
        local_files_only=True,
    )
    _set_only_adapter_trainable(loaded, adapter_name, trainable=trainable)
    return loaded
