# Entropy and KL divergence

## Learning objectives

After this lesson, you should be able to:

- calculate categorical entropy from logits;
- distinguish full-distribution KL from a sampled log-ratio;
- explain why one sampled direct-KL value may be negative;
- recognize the nonnegative sampled k3 estimator;
- explain why NanoPT performs these calculations in FP32.

## Entropy measures uncertainty

For a categorical policy with probabilities $p_v$, Shannon entropy is

$$
H(p) = -\sum_v p_v \log p_v.
$$

A uniform distribution over $V$ choices has entropy $\log V$. A distribution concentrated on one
choice has entropy near zero. For language models, entropy can reveal whether a policy is becoming
more or less uncertain at selected action positions.

[`categorical_entropy`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/entropy.py)
accepts logits with shape `[..., vocabulary]` and returns FP32 entropy with the vocabulary dimension
removed. It uses the complete vocabulary distribution; it is not an entropy proxy based only on
the sampled token.

## Exact categorical KL

KL divergence measures how one distribution differs from another:

$$
D_{\mathrm{KL}}(p \Vert q)
=
\sum_v p_v \left(\log p_v - \log q_v\right).
$$

The order matters: $D_{\mathrm{KL}}(p \Vert q)$ is generally not equal to
$D_{\mathrm{KL}}(q \Vert p)$. Exact KL is nonnegative and equals zero when the distributions match.
[`categorical_kl`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/kl.py) computes this
full sum from policy and reference logits.

## Sampled estimators

When a rollout stores only the sampled action's log probabilities, define

$$
d = \log \pi(a \mid s) - \log \pi_{\mathrm{ref}}(a \mid s).
$$

The direct sampled value is $d$. A single value can be negative even though its expectation under
policy samples is KL. Do not clamp negative direct samples to zero; doing so changes the estimator.

NanoPT also implements the nonnegative k3 estimator:

$$
k_3 = \exp(-d) + d - 1.
$$

The exponential can overflow for an extreme log-ratio. The implementation uses a visible diagnostic
bound and raises rather than silently clamping the formula.

Run the CPU calculation examples with:

```bash
uv run python labs/02_logprob_by_hand.py
```

In addition to selected-token log probabilities, the lab calculates the entropy of a uniform
four-way distribution and exact KL between two two-outcome distributions.

## Common mistakes

- Calling a sampled log-ratio an exact KL.
- Reversing policy and reference in the exact KL formula.
- Averaging over padding or prompt positions instead of applying an action mask.
- Computing large-vocabulary softmax and reductions in BF16.
- Treating parameter distance as behavioral KL.

The hand-computable tests are in
[`test_entropy.py`](https://github.com/shenli/nanopt/blob/main/tests/unit/core/test_entropy.py) and
[`test_kl.py`](https://github.com/shenli/nanopt/blob/main/tests/unit/core/test_kl.py).
