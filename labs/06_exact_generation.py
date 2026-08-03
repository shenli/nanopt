"""CPU lab: inspect one autoregressive sample and compute pass@k by hand."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from nanopt.core.logprobs import causal_token_logps
from nanopt.eval.metrics import pass_at_k
from nanopt.rollout.sampler import SamplingConfig, sample_autoregressive


class TinyTransitionModel(nn.Module):
    """Use the current token ID to select the next-token logits row."""

    def __init__(self) -> None:
        super().__init__()
        # The unused parameter gives the sampler an ordinary model device to inspect.
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "next_token_logits",
            torch.tensor(
                [
                    [0.0, 1.0, 3.0, -1.0],
                    [2.0, 0.0, -1.0, 1.0],
                    [-1.0, 0.0, 1.0, 4.0],
                    [0.0, 5.0, 1.0, -2.0],
                ]
            ),
        )

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        if not torch.equal(attention_mask, torch.ones_like(input_ids)):
            raise ValueError("this tiny fixture expects an unpadded sequence")
        return SimpleNamespace(logits=self.next_token_logits[input_ids] + self.anchor * 0)


def main() -> None:
    model = TinyTransitionModel()
    prompt_ids = torch.tensor([0])
    result = sample_autoregressive(
        model,
        prompt_ids,
        SamplingConfig(max_new_tokens=4, do_sample=False, eos_token_id=3),
    )

    # Token 0 chooses token 2. Token 2 then chooses EOS token 3.
    assert result.generated_token_ids == (2, 3)
    assert result.finish_reason == "eos"

    # Score the exact prompt + sampled IDs in one teacher-forced pass. The sampler's two stored
    # policy log probabilities must match positions 0 and 1 of the causal scorer exactly.
    full_ids = torch.tensor([[*result.prompt_token_ids, *result.generated_token_ids]])
    logits = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids)).logits
    teacher_forced = causal_token_logps(logits, full_ids)[0]
    assert all(
        math.isclose(sampled, scored, rel_tol=0, abs_tol=1e-6)
        for sampled, scored in zip(result.policy_logps, teacher_forced.tolist(), strict=True)
    )

    # Four samples with two correct and k=2 gives 1 - C(2, 2) / C(4, 2) = 5/6.
    estimated_pass_at_2 = pass_at_k(samples=4, correct=2, k=2)
    assert math.isclose(estimated_pass_at_2, 5 / 6)

    print(f"Prompt token IDs:       {result.prompt_token_ids}")
    print(f"Generated token IDs:    {result.generated_token_ids}")
    print(f"Policy log probabilities: {tuple(round(x, 4) for x in result.policy_logps)}")
    print(f"Finish reason:          {result.finish_reason}")
    print(f"Hand-computed pass@2:   {estimated_pass_at_2:.4f}")
    print("Exact generation lab passed.")


if __name__ == "__main__":
    main()
