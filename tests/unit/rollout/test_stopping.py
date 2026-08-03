from __future__ import annotations

import pytest
import torch

from nanopt.rollout.stopping import active_through_eos_mask, first_eos_mask


def test_first_eos_and_active_masks_are_hand_computable() -> None:
    token_ids = torch.tensor([[4, 2, 9, 2], [4, 5, 6, 7]])

    first = first_eos_mask(token_ids, eos_token_id=2)
    active = active_through_eos_mask(token_ids, eos_token_id=2)
    without_eos = active_through_eos_mask(token_ids, eos_token_id=2, include_eos=False)

    assert torch.equal(first, torch.tensor([[False, True, False, False], [False] * 4]))
    assert torch.equal(active, torch.tensor([[True, True, False, False], [True] * 4]))
    assert torch.equal(without_eos, torch.tensor([[True, False, False, False], [True] * 4]))


@pytest.mark.parametrize(
    ("token_ids", "error"),
    [
        (torch.tensor([1, 2]), "shape"),
        (torch.empty((0, 2), dtype=torch.long), "non-empty"),
        (torch.ones((1, 2)), "integer dtype"),
    ],
)
def test_eos_masks_reject_ambiguous_inputs(token_ids: torch.Tensor, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        first_eos_mask(token_ids, 2)


def test_eos_masks_reject_negative_token_id() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        first_eos_mask(torch.tensor([[1, 2]]), -1)
