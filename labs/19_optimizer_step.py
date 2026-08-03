"""Inspect exactly which parameters change in one tiny optimizer step."""

from __future__ import annotations

import torch


def main() -> None:
    """Freeze one parameter, optimize another, and compare before/after values."""

    frozen = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    trainable = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([trainable], lr=0.1)
    frozen_before = frozen.detach().clone()
    trainable_before = trainable.detach().clone()

    prediction = frozen * trainable
    loss = (prediction - 0.0).square().mean()
    loss.backward()
    optimizer.step()

    print(f"Loss:             {loss.item():.1f}")
    print(f"Frozen parameter: {frozen_before.item():.1f} -> {frozen.item():.1f}")
    print(f"Trainable value:  {trainable_before.item():.1f} -> {trainable.item():.1f}")
    assert torch.equal(frozen, frozen_before)
    assert trainable.item() < trainable_before.item()
    print("Optimizer-step lab passed.")


if __name__ == "__main__":
    main()
