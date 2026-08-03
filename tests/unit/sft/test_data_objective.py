from __future__ import annotations

import pytest
import torch

from nanopt.models.renderer import RenderedSupervisedExample
from nanopt.sft.data import CompletionOnlyCollator
from nanopt.sft.objective import completion_only_objective


def _example(tokens: tuple[int, ...], prompt_length: int) -> RenderedSupervisedExample:
    return RenderedSupervisedExample(
        input_ids=tokens,
        attention_mask=(True,) * len(tokens),
        action_mask=(False,) * prompt_length + (True,) * (len(tokens) - prompt_length),
        prompt_length=prompt_length,
        chat_template_sha256="a" * 64,
    )


def test_collator_right_pads_without_activating_prompt_or_padding() -> None:
    collator = CompletionOnlyCollator(pad_token_id=0, max_sequence_length=8)

    batch = collator([_example((1, 2, 3, 4), 2), _example((5, 6, 7), 2)])

    assert batch.input_ids.tolist() == [[1, 2, 3, 4], [5, 6, 7, 0]]
    assert batch.attention_mask.tolist() == [[True] * 4, [True, True, True, False]]
    assert batch.action_mask.tolist() == [
        [False, False, True, True],
        [False, False, True, False],
    ]


def test_completion_loss_is_invariant_to_right_padding() -> None:
    torch.manual_seed(3)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    action_mask = torch.tensor([[False, False, True, True]])
    logits = torch.randn(1, 4, 8)
    expected = completion_only_objective(logits, input_ids, action_mask)

    padded_ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
    padded_mask = torch.tensor([[False, False, True, True, False, False]])
    padded_logits = torch.cat([logits, torch.randn(1, 2, 8)], dim=1)
    actual = completion_only_objective(padded_logits, padded_ids, padded_mask)

    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(actual.token_accuracy, expected.token_accuracy)
    assert actual.active_tokens == expected.active_tokens == 2


def test_prompt_target_logits_receive_zero_gradient() -> None:
    torch.manual_seed(5)
    logits = torch.randn(1, 4, 8, requires_grad=True)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    action_mask = torch.tensor([[False, False, True, True]])

    completion_only_objective(logits, input_ids, action_mask).loss.backward()

    assert logits.grad is not None
    # Row 0 predicts prompt token 2 and row 3 predicts no token; neither belongs to the objective.
    assert torch.count_nonzero(logits.grad[:, 0]).item() == 0
    assert torch.count_nonzero(logits.grad[:, 3]).item() == 0
    # Row 1 is a prompt position, but correctly receives gradient because it predicts the first
    # completion token. Masking targets does not detach the prompt context.
    assert torch.count_nonzero(logits.grad[:, 1]).item() > 0


def test_collator_rejects_truncation_and_empty_completion_targets() -> None:
    collator = CompletionOnlyCollator(pad_token_id=0, max_sequence_length=3)
    with pytest.raises(ValueError, match="does not silently truncate"):
        collator([_example((1, 2, 3, 4), 2)])

    malformed = RenderedSupervisedExample(
        input_ids=(1, 2),
        attention_mask=(True, True),
        action_mask=(False, False),
        prompt_length=1,
        chat_template_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="causally predicted completion"):
        CompletionOnlyCollator(pad_token_id=0, max_sequence_length=3)([malformed])
