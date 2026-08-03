"""Inspect positive- and negative-advantage PPO clipping on CPU."""

from __future__ import annotations

import math

import torch

from nanopt.core.clipping import clipped_policy_loss


def main() -> None:
    """Show why the advantage sign changes which probability-ratio bound matters."""

    old_logps = torch.zeros((1, 2))
    current_logps = torch.tensor([[math.log(1.3), math.log(0.7)]])
    advantages = torch.tensor([[1.0, -1.0]])
    action_mask = torch.ones((1, 2), dtype=torch.bool)
    result = clipped_policy_loss(
        current_logps,
        old_logps,
        advantages,
        action_mask,
        clip_epsilon=0.2,
    )

    print("Ratios:       ", result.ratios.tolist()[0])
    print("Token losses: ", result.per_token_loss.tolist()[0])
    print(f"Clip fraction: {result.clip_fraction.item():.1f}")
    assert torch.allclose(result.ratios, torch.tensor([[1.3, 0.7]]), atol=1e-6)
    assert result.clipped.tolist() == [[True, True]]
    assert torch.allclose(result.loss, torch.tensor(-0.2), atol=1e-6)
    print("PPO-clipping lab passed.")


if __name__ == "__main__":
    main()
