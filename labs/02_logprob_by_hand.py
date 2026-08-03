"""Compare a four-token log-probability calculation with NanoPT's tensor primitive."""

from __future__ import annotations

import math

import torch

from nanopt.core.entropy import categorical_entropy
from nanopt.core.kl import categorical_kl
from nanopt.core.logprobs import causal_token_logps, completion_sequence_logps


def main() -> None:
    """Print selected probabilities, token log probabilities, and their sequence sum."""

    selected_probabilities = [0.5, 0.25, 0.8, 0.1]
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    probabilities = torch.tensor(
        [
            [
                [0.1, 0.5, 0.2, 0.1, 0.1],
                [0.1, 0.1, 0.25, 0.25, 0.3],
                [0.05, 0.05, 0.05, 0.8, 0.05],
                [0.4, 0.2, 0.2, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2, 0.2],
            ]
        ]
    )
    logits = probabilities.log()
    action_mask = torch.tensor([[0, 1, 1, 1, 1]])

    token_logps = causal_token_logps(logits, input_ids)
    sequence_logp = completion_sequence_logps(logits, input_ids, action_mask)
    hand_result = sum(math.log(probability) for probability in selected_probabilities)

    print("Selected probabilities:", selected_probabilities)
    print("Token log probabilities:", [round(value, 4) for value in token_logps[0].tolist()])
    print(f"Hand-computed sum:       {hand_result:.4f}")
    print(f"NanoPT sequence logp:    {sequence_logp.item():.4f}")

    uniform_entropy = categorical_entropy(torch.zeros((1, 4)))
    exact_kl = categorical_kl(
        torch.tensor([[0.75, 0.25]]).log(),
        torch.tensor([[0.5, 0.5]]).log(),
    )
    print(f"Uniform four-way entropy (log 4): {uniform_entropy.item():.4f}")
    print(f"Exact KL([.75,.25] || [.5,.5]):  {exact_kl.item():.4f}")
    print("Log-probability lab passed.")


if __name__ == "__main__":
    main()
