# DPO margins and loss by hand

## Learning objectives

After this lesson, you should be able to:

- distinguish the trainable policy from the frozen reference policy;
- calculate chosen-minus-rejected sequence margins;
- calculate the standard DPO logistic loss;
- predict how changing the chosen or rejected probability changes the loss;
- explain the role of beta without describing it as a learning rate.

## From sequence probabilities to preference margins

For one chosen completion $y_w$ and rejected completion $y_l$, define

$$
\Delta_\theta
=
\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x),
$$

and calculate the same margin for the frozen reference policy:

$$
\Delta_{\mathrm{ref}}
=
\log \pi_{\mathrm{ref}}(y_w \mid x)
-
\log \pi_{\mathrm{ref}}(y_l \mid x).
$$

NanoPT uses masked sums over completion tokens for these sequence log probabilities. Chosen and
rejected sequences must use identical rendering, EOS, masking, and reduction conventions.

## Standard DPO loss

The implicit reward margin and loss are

$$
z = \beta(\Delta_\theta - \Delta_{\mathrm{ref}}),
$$

$$
L_{\mathrm{DPO}} = -\log \sigma(z).
$$

When policy and reference margins match, $z=0$, so the loss is $-\log(0.5)=\log 2$. Increasing the
policy's chosen margin makes $z$ positive and lowers the loss. Making the rejected completion more
likely decreases the margin and raises the loss.

Beta scales the difference from the reference margin. It controls how strongly a given margin
difference affects the logistic objective; it is not the optimizer learning rate.

The canonical implementation in
[`dpo.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/dpo.py) returns the scalar loss
and every intermediate margin so training metrics remain inspectable.

## CPU lab

Run:

```bash
uv run python labs/03_dpo_vertical_slice.py
```

The lab begins with matching policy/reference margins and then increases the chosen policy margin.
No model or dataset download is required.

## Common mistakes

- Reversing chosen and rejected in one margin.
- Comparing raw chosen probabilities instead of chosen-minus-rejected margins.
- Forgetting to subtract the reference margin.
- Using mean sequence log probability in one cache and sum in live policy scoring.
- Allowing gradients into cached or live reference values.

See the sign, beta, gradient, and boundary fixtures in
[`test_dpo.py`](https://github.com/shenli/nanopt/blob/main/tests/unit/core/test_dpo.py).
