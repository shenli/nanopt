"""Tokenizer-chat-template rendering with explicit prompt/completion boundaries."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor


class RendererError(ValueError):
    """Raised when tokenizer rendering cannot preserve NanoPT's token boundary contract."""


class ChatTemplateTokenizer(Protocol):
    """Small tokenizer surface required by :class:`ChatRenderer`."""

    chat_template: str | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_tensors: str,
    ) -> Any: ...


@dataclass(frozen=True)
class RenderedPrompt:
    """Prompt token IDs ready for generation, with no completion actions yet."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    chat_template_sha256: str


@dataclass(frozen=True)
class RenderedSupervisedExample:
    """Full prompt/completion tokens and an action mask in full-token coordinates."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    action_mask: tuple[bool, ...]
    prompt_length: int
    chat_template_sha256: str


def _normalize_token_ids(value: Any) -> tuple[int, ...]:
    """Normalize common tokenizer list/tensor returns without decoding text."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise RendererError("tokenizer mapping output must contain input_ids")
        value = value["input_ids"]
    if isinstance(value, Tensor):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise RendererError(
                f"tokenizer must return one token sequence, got tensor shape {tuple(value.shape)}"
            )
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise RendererError("tokenizer must return one non-empty list of integer token IDs")
    return tuple(value)


def _validate_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if not messages:
        raise RendererError("at least one prompt message is required")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise RendererError(f"message {index} must contain exactly role and content")
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant"}:
            raise RendererError(f"message {index} has unsupported role {role!r}")
        if not content:
            raise RendererError(f"message {index} content must not be empty")
        normalized.append({"role": role, "content": content})
    return normalized


class ChatRenderer:
    """Render prompt and supervised examples using one tokenizer chat template.

    The renderer never locates boundaries in decoded text. It renders the prompt with an assistant
    generation marker, renders the complete conversation separately, and requires the prompt token
    IDs to be an exact prefix of the complete token IDs.
    """

    def __init__(
        self,
        tokenizer: ChatTemplateTokenizer,
        *,
        enable_thinking: bool = False,
        terminal_token_id: int | None = None,
    ) -> None:
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template.strip():
            raise RendererError("tokenizer must provide a non-empty chat_template")
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        if terminal_token_id is not None and terminal_token_id < 0:
            raise RendererError("terminal_token_id must be non-negative")
        self.terminal_token_id = terminal_token_id
        self.chat_template_sha256 = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()

    def _render(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[int, ...]:
        value = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
            return_tensors="pt",
        )
        return _normalize_token_ids(value)

    def render_prompt(self, messages: Sequence[Mapping[str, str]]) -> RenderedPrompt:
        """Render messages ending at the assistant-generation boundary."""

        normalized = _validate_messages(messages)
        if normalized[-1]["role"] == "assistant":
            raise RendererError("generation prompts must not end with an assistant message")
        input_ids = self._render(normalized, add_generation_prompt=True)
        return RenderedPrompt(
            input_ids=input_ids,
            attention_mask=(True,) * len(input_ids),
            chat_template_sha256=self.chat_template_sha256,
        )

    def render_supervised(
        self,
        messages: Sequence[Mapping[str, str]],
        completion: str,
    ) -> RenderedSupervisedExample:
        """Render a completion and prove its exact token boundary against the prompt prefix."""

        if not completion:
            raise RendererError("completion must not be empty")
        normalized = _validate_messages(messages)
        if normalized[-1]["role"] == "assistant":
            raise RendererError(
                "prompt messages must not already contain the target assistant turn"
            )
        prompt_ids = self._render(normalized, add_generation_prompt=True)
        full_messages = [*normalized, {"role": "assistant", "content": completion}]
        full_ids = self._render(full_messages, add_generation_prompt=False)
        if len(full_ids) <= len(prompt_ids):
            raise RendererError("rendered completion must add at least one token after the prompt")
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RendererError(
                "prompt rendering is not an exact prefix of supervised rendering; "
                "the chat template cannot provide a safe completion boundary"
            )
        prompt_length = len(prompt_ids)
        completion_end = len(full_ids)
        if self.terminal_token_id is not None:
            try:
                terminal_index = full_ids.index(self.terminal_token_id, prompt_length)
            except ValueError as exc:
                raise RendererError(
                    "supervised rendering does not contain the configured terminal token"
                ) from exc
            completion_end = terminal_index + 1
        return RenderedSupervisedExample(
            input_ids=full_ids,
            attention_mask=(True,) * len(full_ids),
            action_mask=(
                (False,) * prompt_length
                + (True,) * (completion_end - prompt_length)
                + (False,) * (len(full_ids) - completion_end)
            ),
            prompt_length=prompt_length,
            chat_template_sha256=self.chat_template_sha256,
        )
