"""White-box tensor primitives shared by NanoPT algorithms."""

from nanopt.core.logprobs import causal_token_logps, completion_sequence_logps
from nanopt.core.masks import causal_action_mask, completion_action_mask
from nanopt.core.reductions import masked_mean, masked_sum

__all__ = [
    "causal_action_mask",
    "causal_token_logps",
    "completion_action_mask",
    "completion_sequence_logps",
    "masked_mean",
    "masked_sum",
]
