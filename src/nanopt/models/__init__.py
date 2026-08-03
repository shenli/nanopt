"""Model loading, exact chat rendering, and LoRA adapter lifecycle helpers."""

from nanopt.models.adapters import (
    ParameterCounts,
    attach_lora_adapter,
    clone_lora_adapter,
    freeze_adapter,
    load_lora_adapter,
    parameter_counts,
    save_lora_adapter,
    selected_adapter,
    validate_target_modules,
)
from nanopt.models.loading import LoadedModel, ModelIntegrationError, load_qwen3_base
from nanopt.models.renderer import ChatRenderer, RenderedPrompt, RenderedSupervisedExample

__all__ = [
    "ChatRenderer",
    "LoadedModel",
    "ModelIntegrationError",
    "ParameterCounts",
    "RenderedPrompt",
    "RenderedSupervisedExample",
    "attach_lora_adapter",
    "clone_lora_adapter",
    "freeze_adapter",
    "load_lora_adapter",
    "load_qwen3_base",
    "parameter_counts",
    "save_lora_adapter",
    "selected_adapter",
    "validate_target_modules",
]
