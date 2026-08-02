"""Tiny reductions whose numerators and denominators can be checked by hand."""

from __future__ import annotations

import pytest
import torch

from nanopt.core.reductions import masked_mean, masked_sum


def test_masked_sum_and_mean_use_only_active_values() -> None:
    values = torch.tensor([[1.0, 100.0, 3.0], [4.0, 5.0, 600.0]])
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]])

    assert torch.equal(masked_sum(values, mask, dim=1), torch.tensor([4.0, 9.0]))
    assert torch.equal(masked_mean(values, mask, dim=1), torch.tensor([2.0, 4.5]))
    assert masked_mean(values, mask, dim=(0, 1)).item() == pytest.approx(3.25)


def test_masked_reductions_return_fp32_for_low_precision_inputs() -> None:
    values = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    mask = torch.tensor([[True, True]])

    assert masked_sum(values, mask, dim=1).dtype == torch.float32
    assert masked_mean(values, mask, dim=1).dtype == torch.float32


def test_masked_mean_has_the_expected_gradient() -> None:
    values = torch.tensor([[2.0, 8.0, 4.0]], requires_grad=True)
    mask = torch.tensor([[1, 0, 1]])

    masked_mean(values, mask, dim=1).backward()

    # The active values each contribute one half; the masked middle value contributes nothing.
    assert torch.equal(values.grad, torch.tensor([[0.5, 0.0, 0.5]]))


def test_masked_mean_rejects_any_empty_reduction_slice() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1, 0], [0, 0]])

    with pytest.raises(ValueError, match="zero active values"):
        masked_mean(values, mask, dim=1)


@pytest.mark.parametrize(
    ("values", "mask", "dim", "message"),
    [
        (torch.ones((2, 2)), torch.ones((2, 1)), 1, "identical shapes"),
        (torch.tensor(1.0), torch.tensor(1), 0, "at least one"),
        (
            torch.ones((1, 2)),
            torch.ones((1, 2), dtype=torch.complex64),
            1,
            "real numeric dtype",
        ),
        (torch.ones((1, 2)), torch.tensor([[1.0, float("nan")]]), 1, "finite"),
        (torch.ones((2, 2)), torch.tensor([[1, 2], [1, 0]]), 1, "only 0 and 1"),
        (torch.ones((2, 2)), torch.ones((2, 2)), (), "must not be empty"),
        (torch.ones((2, 2)), torch.ones((2, 2)), 2, "invalid"),
        (torch.ones((2, 2)), torch.ones((2, 2)), (1, -1), "duplicate"),
    ],
)
def test_masked_sum_rejects_invalid_contracts(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int | tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        masked_sum(values, mask, dim)
