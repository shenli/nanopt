"""See why multi-turn Agent SFT trains the next action, not the whole transcript."""

from __future__ import annotations

from typing import Any

from nanopt.models.renderer import ChatRenderer


class TinyChatTokenizer:
    """A character tokenizer that makes every boundary visible without a model download."""

    chat_template = "tiny-chat-v1"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_tensors: str,
    ) -> Any:
        del tokenize, enable_thinking, return_tensors
        role_token = {"system": 10, "user": 20, "assistant": 30}
        tokens = [1]
        for message in conversation:
            tokens.extend([role_token[message["role"]], *message["content"].encode(), 2])
        if add_generation_prompt:
            tokens.append(role_token["assistant"])
        return tokens


messages = [
    {"role": "system", "content": "Return one JSON action."},
    {"role": "user", "content": "Files are unknown."},
    {"role": "assistant", "content": '{"tool":"list_files","arguments":{}}'},
    {"role": "user", "content": "Files: src/fix.py"},
]
target = '{"tool":"read_file","arguments":{"path":"src/fix.py"}}'
example = ChatRenderer(TinyChatTokenizer()).render_supervised(messages, target)

assert not any(example.action_mask[: example.prompt_length])
assert all(example.action_mask[example.prompt_length :])
assert len(example.input_ids) == len(example.attention_mask) == len(example.action_mask)

# The causal objective shifts this full-coordinate mask once. Therefore the first active label is
# predicted by the final prompt token, while the retained list_files action remains context only.
active_targets = sum(example.action_mask[1:])
print("previous action: context only")
print(f"prompt tokens: {example.prompt_length}")
print(f"current action target tokens: {active_targets}")
print("Agent SFT mask invariant passed")
