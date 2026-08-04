# Actor-Critic and PPO clipping

## Learning objectives

After this chapter, you should be able to:

- distinguish policy, value, return, and advantage estimates;
- compute a current-to-old probability ratio;
- explain sign-dependent PPO clipping;
- describe why multiple update epochs create increasing policy mismatch;
- state what NanoPT implements and omits from full actor-critic PPO.

## Actor, critic, and advantage

An actor represents $\pi_\theta(a\mid s)$. A critic estimates $V_\psi(s)$ and can bootstrap a
target. Generalized Advantage Estimation combines temporal-difference residuals across horizons.
NanoPT teaches these concepts but does not train a value model in the v0.1 language-model path.

PPO retains the behavior-policy log probability and forms

$$
\rho_t(\theta)=\exp\left(\log\pi_\theta(a_t\mid s_t)-
\log\pi_{old}(a_t\mid s_t)\right).
$$

The clipped objective maximized per action is

$$
\min\left(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right).
$$

For positive advantage, the upper ratio matters; for negative advantage, the lower ratio matters.
Clipping limits incentive outside the interval but is not a hard KL guarantee.

[`clipped_policy_loss`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/clipping.py)
accepts token log probabilities and action mask with shape `[batch, sequence]`. Advantages may be
response-level `[batch]` or token-level. It reports ratios, clipped tokens, clip fraction, and the
largest absolute log ratio before reducing.

## CPU lab and metrics

```bash
uv run python labs/13_ppo_clipping.py
```

Inspect ratio distribution, clip fraction, approximate/exact KL as defined, entropy, and reward—not
loss alone. A clip fraction of 1.0 in the lab is intentional because both toy ratios cross a bound;
it would be a warning sign if sustained in training.

## NanoPT's boundary

NanoPT's reference GRPO path uses the PPO-style clipped policy term with group-relative advantages
and no learned critic. It stores exact behavior token IDs/log probabilities and performs
synchronous updates. Calling this full actor-critic PPO would be incorrect; calling every clipped
objective identical would also be incorrect because normalization, KL, and advantage definitions
differ.

## Common mistakes, scale mapping, and reading

- Recomputing “old” log probabilities after updating the policy.
- Clipping the ratio before multiplying without testing negative advantages.
- Confusing multiple minibatch epochs with new on-policy data.
- Reporting only the objective name, not normalization and KL conventions.

Production PPO systems add distributed rollout, critic training, GAE, minibatch reuse, and
parameter synchronization. Read the [PPO paper](https://arxiv.org/abs/1707.06347), especially the
surrogate objectives and experimental implementation details.

## Exercises

1. Compute the clipped term for $A=-1$, $\rho=0.7$, and $\epsilon=0.2$.
2. Explain why clipping cannot repair data produced by a very old policy.
3. Compare `token_mean` and `sequence_mean` for responses of lengths 2 and 20.
