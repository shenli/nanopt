"""Inspectable autoregressive sampling primitives."""

from nanopt.rollout.sampler import GenerationResult, SamplingConfig, sample_autoregressive
from nanopt.rollout.stopping import active_through_eos_mask, first_eos_mask

__all__ = [
    "GenerationResult",
    "SamplingConfig",
    "active_through_eos_mask",
    "first_eos_mask",
    "sample_autoregressive",
]
