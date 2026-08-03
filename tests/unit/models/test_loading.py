"""Qwen loader tests replace Hugging Face auto loaders and never use the network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch
from torch import nn

from nanopt.config.loader import ConfigRepository
from nanopt.models import loading
from nanopt.models.loading import ModelIntegrationError, load_qwen3_base


class FakeTokenizer:
    chat_template = "template"
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token_id = 0
    padding_side = "left"

    def __init__(self) -> None:
        self.init_kwargs = {"_commit_hash": "tokenizer-commit"}

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>"
        return 3


class FakeModel(nn.Module):
    def __init__(self, commit: str | None = "model-commit") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.config = SimpleNamespace(_commit_hash=commit, pad_token_id=None)


class FakeAutoTokenizer:
    kwargs: ClassVar[dict[str, Any]] = {}
    tokenizer: ClassVar[Any] = FakeTokenizer()

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
        cls.kwargs = {"model_id": model_id, **kwargs}
        return cls.tokenizer


class FakeAutoModel:
    kwargs: ClassVar[dict[str, Any]] = {}
    model: ClassVar[Any] = FakeModel()

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
        cls.kwargs = {"model_id": model_id, **kwargs}
        return cls.model


@pytest.fixture
def model_profile() -> Any:
    return ConfigRepository().model("qwen3_0_6b_base")


def test_qwen_loader_passes_explicit_safe_options_and_records_revisions(
    monkeypatch: pytest.MonkeyPatch,
    model_profile: Any,
) -> None:
    FakeAutoTokenizer.tokenizer = FakeTokenizer()
    FakeAutoModel.model = FakeModel()
    monkeypatch.setattr(loading, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(loading, "AutoModelForCausalLM", FakeAutoModel)

    result = load_qwen3_base(model_profile, local_files_only=True, device_map="cpu")

    assert result.model_revision == "model-commit"
    assert result.tokenizer_revision == "tokenizer-commit"
    assert result.parameters.total == 2
    assert result.tokenizer.padding_side == "right"
    assert result.model.config.pad_token_id == 0
    assert FakeAutoTokenizer.kwargs["trust_remote_code"] is False
    assert FakeAutoTokenizer.kwargs["local_files_only"] is True
    assert FakeAutoTokenizer.kwargs["revision"] == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    assert FakeAutoModel.kwargs["torch_dtype"] == torch.bfloat16
    assert FakeAutoModel.kwargs["device_map"] == "cpu"
    assert FakeAutoModel.kwargs["use_safetensors"] is True


def test_qwen_loader_rejects_wrong_model_and_old_transformers(
    monkeypatch: pytest.MonkeyPatch,
    model_profile: Any,
) -> None:
    wrong_source = model_profile.source.model_copy(update={"model_id": "other/model"})
    wrong_profile = model_profile.model_copy(update={"source": wrong_source})
    with pytest.raises(ModelIntegrationError, match="requires"):
        load_qwen3_base(wrong_profile, local_files_only=True)

    monkeypatch.setattr(loading.importlib.metadata, "version", lambda _name: "4.0.0")
    with pytest.raises(ModelIntegrationError, match="older than required"):
        load_qwen3_base(model_profile, local_files_only=True)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("chat_template", None, "chat template"),
        ("eos_token_id", None, "eos_token_id"),
    ],
)
def test_qwen_loader_rejects_incomplete_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    model_profile: Any,
    attribute: str,
    value: Any,
    message: str,
) -> None:
    tokenizer = FakeTokenizer()
    setattr(tokenizer, attribute, value)
    FakeAutoTokenizer.tokenizer = tokenizer
    monkeypatch.setattr(loading, "AutoTokenizer", FakeAutoTokenizer)

    with pytest.raises(ModelIntegrationError, match=message):
        load_qwen3_base(model_profile, local_files_only=True)


def test_qwen_loader_requires_resolved_immutable_revisions(
    monkeypatch: pytest.MonkeyPatch,
    model_profile: Any,
) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.init_kwargs = {}
    FakeAutoTokenizer.tokenizer = tokenizer
    FakeAutoModel.model = FakeModel(commit=None)
    unpinned_source = model_profile.source.model_copy(
        update={"revision": None, "tokenizer_revision": None}
    )
    unpinned_profile = model_profile.model_copy(update={"source": unpinned_source})
    monkeypatch.setattr(loading, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(loading, "AutoModelForCausalLM", FakeAutoModel)

    with pytest.raises(ModelIntegrationError, match="immutable revisions"):
        load_qwen3_base(unpinned_profile, local_files_only=True)
