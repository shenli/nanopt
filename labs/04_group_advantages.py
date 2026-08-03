"""Calculate GRPO group advantages and one clipped policy update by hand."""

from __future__ import annotations

import torch

from nanopt.core.advantages import group_relative_advantages
from nanopt.core.clipping import clipped_policy_loss


def main() -> None:
    """Show useful and degenerate reward groups plus positive-advantage clipping."""

    rewards = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    result = group_relative_advantages(rewards, epsilon=1e-8)
    print("Rewards:", rewards.tolist())
    print("Group means:", result.group_mean.tolist())
    print("Group population std:", result.group_std.tolist())
    print("Advantages:", result.advantages.tolist())
    print("Degenerate groups:", result.degenerate_groups.tolist())

    # Ratio 1.5 with positive advantage 1.0 is limited to 1.2 by epsilon=0.2.
    clipped = clipped_policy_loss(
        current_logps=torch.tensor([[0.4054651]]),
        old_logps=torch.tensor([[0.0]]),
        advantages=torch.tensor([1.0]),
        action_mask=torch.tensor([[1]]),
        clip_epsilon=0.2,
    )
    print(f"Clipped positive-advantage loss: {clipped.loss.item():.1f}")
    print(f"Clip fraction:                  {clipped.clip_fraction.item():.1f}")
    print("Group-advantage lab passed.")


if __name__ == "__main__":
    main()
