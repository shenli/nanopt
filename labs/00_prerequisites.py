"""Run a CPU-only readiness check for the concepts used by NanoPT lessons."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def check_environment() -> None:
    """Verify the supported interpreter, project root, and required command-line tool."""

    python_version = sys.version_info[:2]
    if python_version not in {(3, 11), (3, 12)}:
        raise RuntimeError(f"NanoPT requires Python 3.11 or 3.12, found {python_version}")
    if shutil.which("uv") is None:
        raise RuntimeError("uv is not available on PATH")
    if not Path("pyproject.toml").is_file():
        raise RuntimeError("run this lab from the NanoPT repository root")

    print(f"[ok] Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"[ok] PyTorch {torch.__version__}")
    print("[ok] Running from the NanoPT repository root")


def check_tensor_basics() -> None:
    """Demonstrate shape, dtype, indexing, and broadcasting with tiny tensors."""

    token_ids = torch.tensor([[10, 11, 12], [20, 21, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    assert token_ids.shape == (2, 3)
    assert token_ids.dtype == torch.int64
    assert torch.equal(token_ids[:, 1:], torch.tensor([[11, 12], [21, 0]]))

    # Broadcasting expands one scale per batch item across the sequence dimension.
    per_batch_scale = torch.tensor([[1.0], [10.0]])
    scaled_mask = attention_mask.float() * per_batch_scale
    assert torch.equal(scaled_mask, torch.tensor([[1.0, 1.0, 1.0], [10.0, 10.0, 0.0]]))

    print(f"[ok] Tensor shape and dtype: {token_ids.shape}, {token_ids.dtype}")
    print("[ok] Indexing and broadcasting")


def check_probability_basics() -> None:
    """Show that log-softmax creates normalized log probabilities."""

    logits = torch.tensor([[0.0, 1.0, 2.0]])
    log_probs = F.log_softmax(logits, dim=-1)
    probabilities = log_probs.exp()
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))

    # Multiplication in probability space becomes addition in log-probability space.
    direct_probability = torch.tensor(0.5) * torch.tensor(0.8)
    summed_log_probability = torch.log(torch.tensor(0.5)) + torch.log(torch.tensor(0.8))
    torch.testing.assert_close(summed_log_probability.exp(), direct_probability)

    print("[ok] Softmax probabilities sum to one")
    print("[ok] log(a * b) = log(a) + log(b)")


def check_autograd_basics() -> None:
    """Differentiate a one-parameter loss and inspect the resulting gradient."""

    parameter = torch.tensor(1.0, requires_grad=True)
    loss = (parameter - 3.0).square()
    loss.backward()

    # For loss=(w-3)^2, d(loss)/dw=2(w-3), which is -4 when w=1.
    torch.testing.assert_close(parameter.grad, torch.tensor(-4.0))
    print(f"[ok] Autograd gradient: {parameter.grad.item():.1f}")


def check_causal_coordinates() -> None:
    """Identify the target tokens predicted by a causal language model's logits."""

    input_ids = torch.tensor([[10, 11, 12, 13]])
    target_ids = input_ids[:, 1:]
    assert torch.equal(target_ids, torch.tensor([[11, 12, 13]]))
    print("[ok] Causal targets: token positions [1, 2, 3]")


def main() -> None:
    """Run every readiness check and print a clear terminal result."""

    print("NanoPT prerequisite self-check\n")
    check_environment()
    check_tensor_basics()
    check_probability_basics()
    check_autograd_basics()
    check_causal_coordinates()
    print("\nPrerequisite self-check passed.")


if __name__ == "__main__":
    main()
