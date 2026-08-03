# REINFORCE and the score-function gradient

## Learning objectives

After this chapter, you should be able to:

- derive the sign of a one-action REINFORCE update;
- explain why log probability appears in the loss;
- use a baseline to form an advantage without changing the target expectation;
- map one response-level advantage across its active token positions;
- recognize high variance and delayed reward as practical limitations.

## From expected return to a sampled loss

For objective $J(\theta)=\mathbb{E}_{a\sim\pi_\theta}[R(a)]$, the log-derivative identity gives

$$
\nabla_\theta J(\theta)
=
\mathbb{E}\left[R(a)\nabla_\theta\log\pi_\theta(a)\right].
$$

Using advantage $A=R-b$ with an action-independent baseline $b$, a loss for gradient descent is

$$
L_{PG}=-A\log\pi_\theta(a).
$$

If $A>0$, descent raises the sampled action's log probability. If $A<0$, it lowers it. In language
models, sequence log probability is a masked sum of active token log probabilities. Sharing one
terminal advantage across tokens is a modeling decision, not proof that every token deserves equal
credit.

## Hand-computable lab

```bash
uv run python labs/12_reinforce.py
```

Two actions begin at probability 0.5. Action 0 receives reward 1 with baseline 0.25, so its advantage
is 0.75. The exact logit gradient is `[-0.375, 0.375]`: gradient descent increases the selected
logit and decreases the other.

NanoPT's GRPO path uses group-relative advantages rather than a learned value baseline, then applies
a clipped ratio objective rather than plain REINFORCE. The sign intuition remains useful.

## Common mistakes and industrial mapping

- Detaching the selected log probability before multiplying by advantage.
- Backpropagating through a reward or baseline that is meant to be fixed.
- Including prompt/padding positions in the action mask.
- Saying a baseline removes bias without checking its action dependence.
- Reusing old samples as if they came from the current policy.

Larger systems reduce variance with critics, grouped samples, normalized advantages, and larger
batches. They also face stale-policy and long-horizon problems that the two-action lab excludes.

## Primary reading and exercises

Read Williams' [REINFORCE paper](https://link.springer.com/article/10.1007/BF00992696) for the
score-function estimator and the [PPO paper](https://arxiv.org/abs/1707.06347) for the clipped
extension used in the next chapter.

1. Repeat the lab with reward 0 and baseline 0.25; predict the sign first.
2. Show why adding a constant baseline leaves the expected two-action gradient unchanged.
3. Describe what additional record is needed to reuse a sampled token under a ratio objective.
