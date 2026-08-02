"""Inspect NanoPT's causal shift and completion mask on a hand-checkable example."""

from __future__ import annotations

import math

import torch

from nanopt.core.logprobs import causal_token_logps, completion_sequence_logps
from nanopt.core.masks import causal_action_mask, completion_action_mask


def main() -> None:
    """Print token coordinates, shifted masks, and a two-token sequence log probability."""

    token_names = ["<bos>", "Compute", "2+2", "4", "<eos>", "<pad>"]
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    terminal_mask = torch.tensor([[0, 0, 0, 0, 1, 0]])

    # Token position 3 starts the completion. EOS at position 4 is included as an action.
    action_mask = completion_action_mask(
        attention_mask,
        completion_starts=torch.tensor([3]),
        terminal_mask=terminal_mask,
        include_terminal=True,
    )
    prediction_mask = causal_action_mask(action_mask)

    # Each row is a vocabulary distribution at one token position. Position 2 assigns
    # probability 0.5 to completion token ID 3; position 3 assigns 0.8 to EOS token ID 4.
    probabilities = torch.tensor(
        [
            [
                [0.4, 0.2, 0.2, 0.1, 0.1],
                [0.1, 0.2, 0.6, 0.05, 0.05],
                [0.1, 0.1, 0.2, 0.5, 0.1],
                [0.05, 0.05, 0.05, 0.05, 0.8],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
            ]
        ]
    )
    logits = probabilities.log()
    token_logps = causal_token_logps(logits, input_ids)
    sequence_logp = completion_sequence_logps(logits, input_ids, action_mask)

    print("Token positions:       ", list(range(len(token_names))))
    print("Tokens:                ", token_names)
    print("Full-token action mask:", action_mask[0].tolist())
    print("Causal action mask:    ", prediction_mask[0].tolist())
    print("Causal token logps:    ", [round(value, 4) for value in token_logps[0].tolist()])
    print()
    print("Active probabilities:   [0.5, 0.8]")
    print(f"Expected log(0.5)+log(0.8) = {math.log(0.5) + math.log(0.8):.4f}")
    print(f"NanoPT sequence logp          = {sequence_logp.item():.4f}")


if __name__ == "__main__":
    main()
