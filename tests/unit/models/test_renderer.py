"""Renderer boundary tests use a deterministic tokenizer with no external files."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
import torch

from nanopt.models.renderer import ChatRenderer, RendererError


class FakeChatTokenizer:
    chat_template = "<role>{{ role }}</role>{{ content }}"

    def __init__(self, *, tensor_output: bool = False, break_prefix: bool = False) -> None:
        self.tensor_output = tensor_output
        self.break_prefix = break_prefix
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_tensors: str,
    ) -> Any:
        self.calls.append(
            {
                "conversation": conversation,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "return_tensors": return_tensors,
            }
        )
        role_ids = {"system": 10, "user": 20, "assistant": 30}
        ids = [1]
        for message in conversation:
            ids.extend(
                [role_ids[message["role"]], *[100 + ord(char) for char in message["content"]], 2]
            )
        if add_generation_prompt:
            ids.append(role_ids["assistant"])
        if self.break_prefix and conversation[-1]["role"] == "assistant":
            ids[0] = 999
        return torch.tensor([ids]) if self.tensor_output else ids


class TrailingTemplateTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        ids = super().apply_chat_template(*args, **kwargs)
        conversation = args[0]
        if conversation[-1]["role"] == "assistant":
            ids.append(99)
        return ids


def test_render_prompt_records_template_hash_and_thinking_setting() -> None:
    tokenizer = FakeChatTokenizer(tensor_output=True)
    renderer = ChatRenderer(tokenizer, enable_thinking=False)

    result = renderer.render_prompt([{"role": "user", "content": "x"}])

    assert result.input_ids == (1, 20, 220, 2, 30)
    assert result.attention_mask == (True,) * 5
    assert (
        result.chat_template_sha256 == hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    )
    assert tokenizer.calls[0]["tokenize"] is True
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[0]["enable_thinking"] is False
    assert tokenizer.calls[0]["return_tensors"] == "pt"


def test_render_supervised_marks_only_tokens_after_exact_prompt_prefix() -> None:
    renderer = ChatRenderer(FakeChatTokenizer())

    result = renderer.render_supervised([{"role": "user", "content": "x"}], "y")

    assert result.input_ids == (1, 20, 220, 2, 30, 221, 2)
    assert result.prompt_length == 5
    assert result.action_mask == (False, False, False, False, False, True, True)
    assert result.attention_mask == (True,) * 7


def test_render_supervised_excludes_tokens_after_configured_terminal() -> None:
    renderer = ChatRenderer(TrailingTemplateTokenizer(), terminal_token_id=2)

    result = renderer.render_supervised([{"role": "user", "content": "x"}], "y")

    assert result.input_ids[-3:] == (221, 2, 99)
    assert result.action_mask[-3:] == (True, True, False)


def test_renderer_rejects_a_chat_template_without_stable_prefix() -> None:
    renderer = ChatRenderer(FakeChatTokenizer(break_prefix=True))

    with pytest.raises(RendererError, match="not an exact prefix"):
        renderer.render_supervised([{"role": "user", "content": "x"}], "y")


@pytest.mark.parametrize(
    ("messages", "message"),
    [
        ([], "at least one"),
        ([{"role": "user", "content": "x", "extra": "y"}], "exactly role and content"),
        ([{"role": "tool", "content": "x"}], "unsupported role"),
        ([{"role": "user", "content": ""}], "must not be empty"),
        ([{"role": "assistant", "content": "x"}], "must not end with an assistant"),
    ],
)
def test_render_prompt_rejects_invalid_messages(
    messages: list[dict[str, str]],
    message: str,
) -> None:
    renderer = ChatRenderer(FakeChatTokenizer())

    with pytest.raises(RendererError, match=message):
        renderer.render_prompt(messages)


def test_render_supervised_rejects_empty_or_preexisting_assistant_completion() -> None:
    renderer = ChatRenderer(FakeChatTokenizer())
    prompt = [{"role": "user", "content": "x"}]

    with pytest.raises(RendererError, match="completion must not be empty"):
        renderer.render_supervised(prompt, "")
    with pytest.raises(RendererError, match="must not already contain"):
        renderer.render_supervised([{"role": "assistant", "content": "x"}], "y")


def test_renderer_rejects_missing_template_and_invalid_token_output() -> None:
    tokenizer = FakeChatTokenizer()
    tokenizer.chat_template = None  # type: ignore[assignment]
    with pytest.raises(RendererError, match="non-empty chat_template"):
        ChatRenderer(tokenizer)

    valid = FakeChatTokenizer()
    valid.apply_chat_template = lambda *args, **kwargs: []  # type: ignore[method-assign]
    with pytest.raises(RendererError, match="non-empty list"):
        ChatRenderer(valid).render_prompt([{"role": "user", "content": "x"}])
