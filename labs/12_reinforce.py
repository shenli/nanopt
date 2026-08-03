"""Differentiate one hand-computable REINFORCE update on CPU."""

from __future__ import annotations

import torch


def main() -> None:
    """Increase the selected action probability when its advantage is positive."""

    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    action = 0
    reward = 1.0
    baseline = 0.25
    advantage = reward - baseline
    log_probability = logits.log_softmax(dim=-1)[action]
    loss = -advantage * log_probability
    loss.backward()

    expected_gradient = torch.tensor([-0.375, 0.375])
    print(f"Action probability: {log_probability.exp().item():.2f}")
    print(f"Advantage:          {advantage:.2f}")
    print(f"Gradient:           {logits.grad.tolist() if logits.grad is not None else None}")
    assert logits.grad is not None
    assert torch.allclose(logits.grad, expected_gradient)
    print("REINFORCE lab passed.")


if __name__ == "__main__":
    main()
