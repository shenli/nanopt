"""Inspect DPO margins and loss without loading a language model."""

from __future__ import annotations

import math

import torch

from nanopt.core.dpo import dpo_loss


def main() -> None:
    """Compare matching and improved policy margins against one frozen reference pair."""

    reference_chosen = torch.tensor([-2.0])
    reference_rejected = torch.tensor([-3.0])
    matching = dpo_loss(
        torch.tensor([-2.0]),
        torch.tensor([-3.0]),
        reference_chosen,
        reference_rejected,
        beta=1.0,
    )
    improved = dpo_loss(
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        reference_chosen,
        reference_rejected,
        beta=1.0,
    )

    print(f"Matching margin loss (log 2): {matching.loss.item():.4f} ({math.log(2):.4f})")
    print(f"Improved policy margin:       {improved.policy_margin.item():.1f}")
    print(f"Reference margin:             {improved.reference_margin.item():.1f}")
    print(f"Implicit reward margin:       {improved.implicit_reward_margin.item():.1f}")
    print(f"Improved margin loss:         {improved.loss.item():.4f}")
    print("DPO lab passed.")


if __name__ == "__main__":
    main()
