# Reinforcement-learning foundations for language models

## Learning objectives

After this chapter, you should be able to:

- define state, observation, action, transition, trajectory, reward, and return;
- distinguish Monte Carlo return from a bootstrapped target;
- explain on-policy versus off-policy data in model-version terms;
- map token generation and tool use into the same environment vocabulary;
- identify long-horizon credit assignment as a separate problem from verification.

## The minimal interaction model

At time $t$, an environment has state $s_t$ and exposes observation $o_t$. A policy samples action
$a_t \sim \pi_\theta(\cdot\mid o_t)$. The environment transitions and emits reward $r_t$. A
trajectory is the ordered record of these quantities until termination or truncation.

The discounted return from time $t$ is

$$
G_t = \sum_{k=t}^{T}\gamma^{k-t}r_k,
$$

where $\gamma \in [0,1]$. NanoPT's short verifiable tasks normally use terminal outcome reward and
effectively $\gamma=1$. This does not solve credit assignment: the same final reward may be attached
to many token or tool actions.

## Language models as policies

For single-turn generation, the observation is the prompt and preceding tokens; each next-token ID
is an action. For MiniSWE, a model generates many token actions that decode to one structured tool
action. The environment state also includes workspace files and budgets that are not all reproduced
inside the model.

Monte Carlo methods wait for observed returns. Temporal-difference methods bootstrap from a value
estimate such as $r_t + \gamma V(s_{t+1})$. A baseline can reduce variance without changing the
expected policy gradient when it does not depend on the sampled action.

## Policy version is part of data identity

On-policy data comes from the policy distribution the update claims to optimize. Stored old log
probabilities make one behavior policy explicit. If rollout workers finish after several parameter
updates, the result is stale. Importance ratios can correct limited mismatch, but they do not grant
unlimited reuse or erase support/variance problems.

The reference GRPO path is synchronous: grouped completions and old log probabilities are frozen
before the update. The initial agent-environment validation records exact model token IDs but
performs no policy update.

## Hands-on path

Run the REINFORCE and scheduler labs after this chapter:

```bash
uv run python labs/12_reinforce.py
uv run python labs/15_rollout_scheduler.py
```

The first isolates the score-function gradient. The second shows that long-tail rollouts can become
stale or force partial-work discard.

## Common mistakes, scale mapping, and reading

- Calling a prompt the complete state when external tools/files also matter.
- Treating truncation by a budget as successful termination.
- Mixing policy versions without recording behavior identity.
- Assuming a terminal verifier identifies which earlier action caused success.

At scale, schedulers, replay buffers, critics, and distributed workers elaborate these objects but
do not remove them. Sutton and Barto's
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) is the
background text; use its notation to check whether a report means return, advantage, or reward.

## Exercises

1. Write the state and observation for one `read_file` MiniSWE turn.
2. Compute $G_0$ for rewards `[0, 0, 1]` with $\gamma=1$ and $\gamma=0.9$.
3. Explain why exact verification does not provide token-level credit assignment.
