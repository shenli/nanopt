"""Hand-computable categorical entropy tests."""

from __future__ import annotations

import math

import pytest
import torch

from nanopt.core.entropy import categorical_entropy


def test_uniform_distribution_has_log_vocabulary_entropy() -> None:
    logits = torch.zeros((2, 4))

    result = categorical_entropy(logits)

    assert torch.allclose(result, torch.full((2,), math.log(4.0)))


def test_entropy_runs_in_fp32_and_preserves_gradients() -> None:
    logits = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16, requires_grad=True)

    result = categorical_entropy(logits)
    result.sum().backward()

    assert result.dtype == torch.float32
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (torch.tensor(1.0), "at least a vocabulary"),
        (torch.empty((1, 0)), "must not be empty"),
        (torch.ones((1, 2), dtype=torch.long), "floating-point dtype"),
        (torch.tensor([[0.0, float("nan")]]), "finite"),
    ],
)
def test_entropy_rejects_invalid_logits(logits: torch.Tensor, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        categorical_entropy(logits)
