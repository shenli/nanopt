from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanopt.core.logprobs import causal_token_logps
from nanopt.rollout.sampler import SamplingConfig, sample_autoregressive


class TransitionModel(nn.Module):
    """Tiny causal fixture: the final input ID selects the next-token logits row."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "transitions",
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
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        logits = self.transitions[input_ids] + self.anchor * 0
        return SimpleNamespace(logits=logits)


def test_greedy_sampler_captures_ids_logps_and_eos() -> None:
    model = TransitionModel()
    result = sample_autoregressive(
        model,
        torch.tensor([0]),
        SamplingConfig(max_new_tokens=4, do_sample=False, eos_token_id=3),
    )

    # ID 0 -> greedy ID 2, then ID 2 -> greedy EOS ID 3.
    assert result.generated_token_ids == (2, 3)
    assert result.active_mask == (True, True)
    assert result.finish_reason == "eos"
    assert result.policy_logps == pytest.approx(result.behavior_logps)


def test_sampler_logps_match_teacher_forced_scoring() -> None:
    model = TransitionModel()
    result = sample_autoregressive(
        model,
        torch.tensor([0]),
        SamplingConfig(max_new_tokens=2, do_sample=False),
    )
    full_ids = torch.tensor([[*result.prompt_token_ids, *result.generated_token_ids]])
    teacher_forced_logits = model(
        input_ids=full_ids, attention_mask=torch.ones_like(full_ids)
    ).logits
    teacher_forced = causal_token_logps(teacher_forced_logits, full_ids)[0]

    assert result.policy_logps == pytest.approx(teacher_forced.tolist())


def test_sampler_stops_after_complete_multi_token_protocol_sequence() -> None:
    result = sample_autoregressive(
        TransitionModel(),
        torch.tensor([0]),
        SamplingConfig(
            max_new_tokens=4,
            do_sample=False,
            stop_token_sequences=((2, 3),),
        ),
    )

    assert result.generated_token_ids == (2, 3)
    assert result.finish_reason == "stop_sequence"


def test_sampled_mode_uses_private_deterministic_seed() -> None:
    config = SamplingConfig(max_new_tokens=6, do_sample=True, temperature=1.3, top_p=0.9)
    first = sample_autoregressive(TransitionModel(), torch.tensor([1]), config, seed=123)
    torch.manual_seed(9999)
    second = sample_autoregressive(TransitionModel(), torch.tensor([1]), config, seed=123)

    assert first == second
    assert len(first.generated_token_ids) == 6
    assert first.finish_reason == "length"


def test_sampling_records_behavior_distribution_separately() -> None:
    result = sample_autoregressive(
        TransitionModel(),
        torch.tensor([0]),
        SamplingConfig(max_new_tokens=1, do_sample=True, temperature=0.5, top_p=0.8),
        seed=7,
    )
    assert result.policy_logps[0] != pytest.approx(result.behavior_logps[0])


def test_sampler_restores_training_mode() -> None:
    model = TransitionModel()
    model.train()
    sample_autoregressive(model, torch.tensor([0]), SamplingConfig(1, False))
    assert model.training


def test_sampler_rejects_nonfinite_next_token_logits() -> None:
    model = TransitionModel()
    model.transitions[0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        sample_autoregressive(model, torch.tensor([0]), SamplingConfig(1, False))


@pytest.mark.parametrize(
    "config",
    [
        lambda: SamplingConfig(0, False),
        lambda: SamplingConfig(1, True, temperature=0),
        lambda: SamplingConfig(1, True, top_p=0),
        lambda: SamplingConfig(1, False, eos_token_id=-1),
        lambda: SamplingConfig(1, False, stop_token_sequences=((),)),
        lambda: SamplingConfig(1, False, stop_token_sequences=((-1,),)),
    ],
)
def test_sampling_config_rejects_invalid_values(config: object) -> None:
    with pytest.raises(ValueError):
        config()  # type: ignore[operator]


def test_sampler_rejects_invalid_prompt_contract() -> None:
    model = TransitionModel()
    config = SamplingConfig(1, False)
    with pytest.raises(ValueError, match="shape"):
        sample_autoregressive(model, torch.tensor([[0]]), config)
    with pytest.raises(TypeError, match="dtype"):
        sample_autoregressive(model, torch.tensor([0.0]), config)
