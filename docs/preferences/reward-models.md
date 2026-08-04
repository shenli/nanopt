# Reward models: pairwise ranking and its limits

## Learning objectives

After this chapter, you should be able to:

- calculate a pairwise logistic ranking loss;
- distinguish a learned reward from an exact verifier;
- explain why reward scale and calibration matter;
- identify shortcut features and distribution shift;
- state why NanoPT's golden path does not train a reward model.

## From comparisons to a scalar

Suppose a reward model assigns scalar scores $r_\phi(x,y^+)$ and $r_\phi(x,y^-)$ to chosen and
rejected responses for prompt $x$. The Bradley–Terry logistic loss is

$$
L_{RM} = -\log \sigma\left(r_\phi(x,y^+) - r_\phi(x,y^-)\right),
$$

where $\sigma$ is the sigmoid. A zero margin gives $-\log(0.5) \approx 0.693$. A positive margin
lowers the loss; swapping the pair raises it. For a batch, the per-pair losses have shape `[batch]`
before reduction to a scalar.

Run the hand calculation:

```bash
uv run python labs/11_reward_ranking.py
```

## Score is not truth

A reward model estimates preferences represented in its training data. It can exploit response
length, style, familiar phrases, or annotator artifacts. Its scalar may be poorly calibrated outside
the comparison distribution. Outcome reward asks whether the final result succeeded; process reward
scores intermediate work. Both can be learned or programmatic, and both can be wrong.

NanoPT uses exact arithmetic AST verification in the required path. This deliberately removes a
learned-reward variable while teaching rollout and policy optimization. DPO uses preference pairs
directly without fitting a separate scalar reward model. A reward-model experiment is a documented
extension, not an implicit part of the DPO or GRPO reference paths.

## Audit questions

- Are chosen/rejected positions balanced for length and formatting?
- Does held-out ranking accuracy survive a task-family shift?
- Are ties and annotator disagreement retained or forced into false certainty?
- Does reward scale drift between model versions?
- Can an adversarial response raise score without improving the real outcome?

## Common mistakes and scale mapping

- Interpreting high pairwise accuracy as calibrated reward.
- Training and evaluating on pairs derived by the same shortcut generator.
- Combining heterogeneous reward components without reporting their weights.
- Optimizing a proxy repeatedly without a fixed outcome evaluation.

Production systems use larger ensembles, calibration sets, safety-specific models, and ongoing red
teaming. Those controls change scale, not the basic requirement that the reward boundary remain
separate from the policy being optimized.

## Primary reading and exercises

The [InstructGPT paper](https://arxiv.org/abs/2203.02155) presents a well-known learned preference
model and PPO pipeline. Read its reward-model data and evaluation sections before its headline
policy results.

1. Compute the loss for margins `0`, `1`, and `-1`.
2. Construct two responses where length predicts the chosen label for the wrong reason.
3. Explain why an exact verifier can still be hacked even though it is not learned.
