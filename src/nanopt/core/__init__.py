"""White-box tensor primitives shared by NanoPT algorithms."""

from nanopt.core.advantages import GroupAdvantageResult, group_relative_advantages
from nanopt.core.clipping import ClippedPolicyLossResult, clipped_policy_loss, probability_ratio
from nanopt.core.dpo import DpoLossResult, dpo_loss, preference_margin
from nanopt.core.entropy import categorical_entropy
from nanopt.core.kl import categorical_kl, sampled_direct_kl, sampled_k3_kl
from nanopt.core.logprobs import causal_token_logps, completion_sequence_logps
from nanopt.core.masks import causal_action_mask, completion_action_mask
from nanopt.core.reductions import masked_mean, masked_sum

__all__ = [
    "ClippedPolicyLossResult",
    "DpoLossResult",
    "GroupAdvantageResult",
    "categorical_entropy",
    "categorical_kl",
    "causal_action_mask",
    "causal_token_logps",
    "clipped_policy_loss",
    "completion_action_mask",
    "completion_sequence_logps",
    "dpo_loss",
    "group_relative_advantages",
    "masked_mean",
    "masked_sum",
    "preference_margin",
    "probability_ratio",
    "sampled_direct_kl",
    "sampled_k3_kl",
]
