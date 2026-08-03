"""Calculate one pairwise reward-model ranking loss by hand on CPU."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def main() -> None:
    """Show that increasing the chosen-minus-rejected margin lowers ranking loss."""

    chosen_score = torch.tensor([2.0])
    rejected_score = torch.tensor([1.0])
    margin = chosen_score - rejected_score
    loss = -F.logsigmoid(margin)
    wider_loss = -F.logsigmoid(margin + 1.0)

    print(f"Reward margin:      {margin.item():.1f}")
    print(f"Pairwise loss:      {loss.item():.4f}")
    print(f"Wider-margin loss: {wider_loss.item():.4f}")
    assert torch.allclose(loss, torch.tensor([0.3132617]), atol=1e-6)
    assert wider_loss.item() < loss.item()
    print("Reward-ranking lab passed.")


if __name__ == "__main__":
    main()
