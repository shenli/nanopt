# Reading guide: objectives from preference ranking to GRPO

## Learning objectives

After this guide, you should be able to:

- locate the assumptions behind DPO, PPO, and GRPO rather than comparing names;
- identify the behavior/reference policies and reduction used by each objective;
- separate a paper's mathematical contribution from its data and systems contribution;
- map each primary source to NanoPT code and explicit omissions.

## Fixed reading questions

For every objective, write down: sampled data source, optimized policy, comparison/reference policy,
advantage or preference signal, ratio/KL term, masking, reduction, update epochs, and freshness
assumption. If the paper does not specify an implementation detail, mark it **unknown**.

## DPO

Read [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), focusing on the
KL-regularized reward-model derivation and binary preference likelihood. NanoPT implements the
policy/reference chosen-minus-rejected margin in `src/nanopt/core/dpo.py`, caches frozen-reference
sequence log probabilities, and uses controlled synthetic pairs. It omits human preference
collection, a learned reward model, and broad assistant evaluation.

Questions:

1. Where does the reference policy enter the implicit reward?
2. Which assumptions connect the preference likelihood to the policy ratio?
3. How would changing sequence-log-probability normalization change length behavior?

## PPO

Read [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), especially the
clipped surrogate, multiple epochs, and actor-critic experimental setup. NanoPT implements the
token-level probability ratio and clipped policy term with explicit normalization. It does not
implement a learned critic/GAE pipeline for the language model.

Questions:

1. Is clipping a hard policy-distance constraint?
2. What becomes increasingly off-policy across update epochs?
3. Which metrics reveal an update much larger than intended?

## GRPO

Read [DeepSeekMath](https://arxiv.org/abs/2402.03300) for the original group-relative method, then
[DeepSeek-R1](https://arxiv.org/abs/2501.12948) for a larger multi-stage reasoning-RL pipeline.
NanoPT implements synchronous grouped completions, population-standardized group advantages,
exact-token old log probabilities, clipping, optional KL, and named reductions. It does not claim
its small arithmetic recipe reproduces either model or data pipeline.

Questions:

1. What replaces the learned value baseline?
2. What happens when all group rewards are equal?
3. Which choices called “GRPO” remain implementation variants?

## Comparative exercise

Build a three-row table with one exact formula and one exact data-freshness assumption per method.
Do not write “PPO-like” or “same as GRPO” without naming the differing advantage, KL, and reduction.
