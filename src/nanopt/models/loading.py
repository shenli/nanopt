"""Explicit Qwen3 0.6B Base model and tokenizer loading."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import torch
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanopt.config.models import ModelProfile
from nanopt.models.adapters import ParameterCounts, parameter_counts

QWEN3_BASE_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
QWEN_CHAT_TERMINATOR = "<|im_end|>"


class ModelIntegrationError(RuntimeError):
    """Raised when a model/tokenizer violates the recorded integration contract."""


@dataclass(frozen=True)
class LoadedModel:
    """Loaded objects plus immutable revisions and parameter-count evidence."""

    model: Any
    tokenizer: Any
    model_revision: str
    tokenizer_revision: str
    parameters: ParameterCounts


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[name]
    except KeyError as exc:
        raise ModelIntegrationError(f"unsupported model dtype: {name}") from exc


def _resolved_revision(value: Any, fallback: str | None) -> str | None:
    commit = getattr(value, "_commit_hash", None)
    if isinstance(commit, str) and commit:
        return commit
    init_kwargs = getattr(value, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        commit = init_kwargs.get("_commit_hash")
        if isinstance(commit, str) and commit:
            return commit
    return fallback


def _validate_profile(profile: ModelProfile) -> None:
    if profile.source.model_id != QWEN3_BASE_MODEL_ID:
        raise ModelIntegrationError(
            f"Qwen3 base loader requires {QWEN3_BASE_MODEL_ID!r}, got {profile.source.model_id!r}"
        )
    if profile.source.trust_remote_code:
        raise ModelIntegrationError("trust_remote_code must remain false")
    if profile.checks is not None:
        installed = Version(importlib.metadata.version("transformers"))
        minimum = Version(profile.checks.min_transformers_version)
        if installed < minimum:
            raise ModelIntegrationError(
                f"transformers {installed} is older than required version {minimum}"
            )


def _validate_tokenizer(tokenizer: Any) -> None:
    if not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise ModelIntegrationError("tokenizer does not provide a chat template")
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ModelIntegrationError("tokenizer does not define eos_token_id")
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ModelIntegrationError("tokenizer cannot derive padding from an EOS token")
        tokenizer.pad_token = eos_token
    tokenizer.padding_side = "right"
    qwen_chat_terminator_id(tokenizer)


def qwen_chat_terminator_id(tokenizer: Any) -> int:
    """Return the special token that ends a Qwen chat-template assistant turn.

    Qwen's generic tokenizer EOS is ``<|endoftext|>``, while rendered chat turns end with
    ``<|im_end|>``. Generation must stop on the latter to match the exact SFT target boundary.
    """

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise ModelIntegrationError("tokenizer cannot resolve the Qwen chat terminator")
    token_id = convert(QWEN_CHAT_TERMINATOR)
    if not isinstance(token_id, int) or token_id < 0:
        raise ModelIntegrationError("tokenizer returned an invalid Qwen chat terminator ID")
    return token_id


def load_qwen3_base(
    profile: ModelProfile,
    *,
    local_files_only: bool = False,
    device_map: str | dict[str, Any] | None = None,
) -> LoadedModel:
    """Load the configured Qwen3 base model and resolve immutable source revisions.

    This is the only M2 function allowed to contact the Hugging Face Hub. Tests pass
    ``local_files_only=True`` or replace the auto loaders. The returned revisions come from Hub
    commit metadata when the profile uses a moving name; failure to resolve them is an error because
    a reproducible run cannot depend on an unrecorded model snapshot.
    """

    _validate_profile(profile)
    source = profile.source
    tokenizer_revision = source.tokenizer_revision or source.revision
    tokenizer = AutoTokenizer.from_pretrained(
        source.model_id,
        revision=tokenizer_revision,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    _validate_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        source.model_id,
        revision=source.revision,
        trust_remote_code=False,
        local_files_only=local_files_only,
        torch_dtype=_torch_dtype(profile.loading.dtype),
        low_cpu_mem_usage=profile.loading.low_cpu_mem_usage,
        use_safetensors=profile.loading.use_safetensors,
        device_map=device_map,
    )
    model_commit = _resolved_revision(getattr(model, "config", model), source.revision)
    tokenizer_commit = _resolved_revision(tokenizer, tokenizer_revision)
    if model_commit is None or tokenizer_commit is None:
        raise ModelIntegrationError(
            "model and tokenizer must resolve to immutable revisions; pin revisions in the profile "
            "or load from a source that reports its commit hash"
        )
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_revision=model_commit,
        tokenizer_revision=tokenizer_commit,
        parameters=parameter_counts(model),
    )
