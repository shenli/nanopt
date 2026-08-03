"""Opt-in check against the pinned real Qwen tokenizer; excluded from normal CPU validation."""

from __future__ import annotations

import os

import pytest
from transformers import AutoTokenizer

from nanopt.config.loader import ConfigRepository
from nanopt.models.renderer import ChatRenderer

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("NANOPT_RUN_NETWORK_TESTS") != "1",
        reason="set NANOPT_RUN_NETWORK_TESTS=1 to access the pinned Hugging Face tokenizer",
    ),
]


def test_pinned_qwen_tokenizer_preserves_supervised_prefix_boundary() -> None:
    profile = ConfigRepository().model("qwen3_0_6b_base")
    revision = profile.source.tokenizer_revision
    assert revision is not None
    tokenizer = AutoTokenizer.from_pretrained(
        profile.source.model_id,
        revision=revision,
        trust_remote_code=False,
    )
    renderer = ChatRenderer(tokenizer, enable_thinking=profile.renderer.enable_thinking)

    rendered = renderer.render_supervised(
        [{"role": "user", "content": "Compute 2+2."}],
        "<answer>4</answer>",
    )

    assert rendered.prompt_length > 0
    assert any(rendered.action_mask)
    assert not any(rendered.action_mask[: rendered.prompt_length])
    assert tokenizer.eos_token_id is not None
    assert tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=True) == ""
