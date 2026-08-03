# Group advantages and policy clipping

## Learning objectives

After this lesson, you should be able to:

- calculate centered and population-standardized rewards within a prompt group;
- identify an all-equal reward group as degenerate;
- calculate a current-to-old policy probability ratio;
- explain why clipping behaves differently for positive and negative advantages;
- distinguish token-mean from sequence-mean loss normalization.

## Group-relative advantages

For $G$ completions sampled from one prompt, let their rewards be $r_1,\ldots,r_G$. The reference
`group_zscore` mode calculates

$$
\mu = \frac{1}{G}\sum_g r_g,
$$

$$
\sigma = \sqrt{\frac{1}{G}\sum_g(r_g-\mu)^2},
$$

$$
A_g = \frac{r_g-\mu}{\sigma+\epsilon_A}.
$$

The standard deviation is the population value (`unbiased=False` in PyTorch). For rewards
`[1, 2, 3]`, the mean is 2 and population standard deviation is $\sqrt{2/3}$. The advantages are
approximately `[-1.2247, 0, 1.2247]` and sum to zero.

If all group rewards match, every centered reward and advantage is exactly zero. That group supplies
no policy-gradient signal and should be counted as degenerate. The alternative `group_centered`
mode subtracts the mean without standard-deviation scaling; the two modes weight groups differently.

## Probability ratios and clipping

For one stored action,

$$
\rho = \exp(\log \pi_\theta - \log \pi_{\mathrm{old}}).
$$

If the policy has not changed, the ratio is one. The clipped per-token loss is

$$
\ell
=
-\min\left(
\rho A,
\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A
\right).
$$

For positive advantage, a ratio above $1+\epsilon$ is clipped because the update is trying to make a
good action too much more likely. For negative advantage, a ratio below $1-\epsilon$ is clipped
because the update is trying to make a bad action too much less likely. Multiplying by a negative
advantage reverses the ordering inside `min`; this is why sign-specific tests are essential.

## Loss normalization

`token_mean` gives every active token equal weight across the rollout batch. Longer responses
therefore contribute more terms. `sequence_mean` first averages active tokens inside each response,
then gives each response equal weight. Neither convention is universally identical or harmless, so
NanoPT names the selected mode in configuration and metrics.

Implementations:

- [`advantages.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/advantages.py)
- [`clipping.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/clipping.py)

## CPU lab

Run:

```bash
uv run python labs/04_group_advantages.py
```

The lab compares a useful reward group with an all-equal group, then calculates a clipped positive-
advantage update.

Tests cover population standard deviation, degenerate groups, ratio identity, both advantage signs,
unequal response lengths, gradients, and invalid numerical inputs:

- [`test_advantages.py`](https://github.com/shenli/nanopt/blob/main/tests/unit/core/test_advantages.py)
- [`test_clipping.py`](https://github.com/shenli/nanopt/blob/main/tests/unit/core/test_clipping.py)
