# Mini Agent RL: outcomes, exact actions, and policy age

Agent SFT learns from trusted actions. Agent RL instead samples actions from the current policy,
lets those actions change a resettable workspace, and learns from a terminal hidden-verifier
outcome. That extra environment state makes the data contract more important than the optimizer
name.

## Learning objectives

After this chapter, you should be able to:

- explain why a rollout group must begin from identical task snapshots;
- locate old behavior log probabilities in causal token coordinates;
- distinguish terminal outcome reward from token-level credit assignment;
- explain why NanoPT trains on fresh groups and measures stale groups without reusing them;
- interpret a tool-budget study without reducing partial verifier credit to pass/fail.

## One group, one starting state

For task snapshot $s_0$, NanoPT independently resets $G$ workspaces and samples episodes
$\tau_1,\ldots,\tau_G$. It does not fork an already-mutated workspace. The hidden verifier runs
only after each episode terminates and returns outcome reward $R_i \in [0,1]$.

For group-z-scored advantages,

$$
A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon},
\qquad
\mu_R = \frac{1}{G}\sum_{j=1}^{G}R_j.
$$

Equal-reward groups produce zero advantages and remain visible as degenerate groups. NanoPT does
not add an undocumented shaping reward merely to manufacture a gradient.

## Exact multi-turn action coordinates

Every turn is stored separately because its prompt contains a different observation. For action
turn $t$, the record retains:

- the exact online `prompt_token_ids`;
- exact `sampled_token_ids` and an aligned action mask;
- FP32 behavior log probabilities from temperature-one, untruncated sampling;
- optional log probabilities from the frozen Agent SFT reference;
- parse status and tool name for diagnostics.

The collator concatenates the stored prompt and action IDs. If the prompt length is $P$, causal
prediction coordinate $P-1$ scores the first sampled action token. Decoded JSON is never rendered
or tokenized again for the update.

The implementation is deliberately separated by responsibility:

1. [`rl_rollout.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/rl_rollout.py)
   owns reset, sampling seeds, hidden outcome collection, and grouping.
2. [`rl_records.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/rl_records.py)
   rejects mixed snapshots and drifting token coordinates.
3. [`rl_trainer.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/rl_trainer.py)
   exposes collation, reference scoring, clipping, KL, and the optimizer step.
4. [`rl_run.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/rl_run.py)
   alternates fresh collection and updates, then builds the studies and report.

## Short-horizon clipped update

The current/reference policy ratio for sampled token $k$ is

$$
r_k(\theta)=\exp\left(\log\pi_\theta(a_k\mid h_k)
-\log\pi_{\text{old}}(a_k\mid h_k)\right).
$$

The episode advantage is broadcast to every current-action token, never to observations or prior
actions. NanoPT minimizes the masked PPO-style clipped loss plus sampled reference KL. The config
names the advantage, clipping, normalization, KL estimator, and coefficient independently; “Agent
GRPO” is not treated as one universal formula.

Only groups whose `collected_policy_version` equals the current version may update the model.
`max_policy_lag: 0` is validated in config and checked again at the trainer boundary.

## Fresh versus stale is an experiment, not a slogan

After training, the final policy scores the newest and oldest retained exact action IDs. For each
set the report records policy lag, mean and maximum absolute log-ratio drift, and an approximate
importance-weight effective-sample-size fraction. Both points have `used_for_update: false`.

The newest point still has lag one because it is scored after its update. The oldest point spans
all policy versions. This measures drift without silently turning v0.3 into an off-policy replay
algorithm.

## Credit and tool-budget studies

The reference update gives the terminal group-relative advantage to every sampled action token.
The credit study also counts how many tokens would receive credit under a terminal-action-only
ablation. That ablation cannot teach earlier inspection or editing decisions; it is evidence about
coverage, not a second trained checkpoint.

The tool-budget study evaluates the frozen Agent SFT parent and final Agent RL adapter with the
same held-out task at two caps. It reports continuous hidden reward and action validity. A partially
correct repository can earn partial hidden credit, so “not solved” does not necessarily mean zero.

Run the hand-checkable lesson:

```bash
uv run python labs/21_agent_rl_credit.py
```

The tiny lab proves snapshot identity, advantage signs, and all-action versus terminal-only token
coverage without downloading a model.

## Run the reference path

Start from the adapter produced by v0.2:

```bash
uv run nanopt train agent-rl \
  --agent-sft-adapter artifacts/runs/AGENT_SFT_RUN/adapter/agent_sft \
  --tasks-root tasks/mini_swe_v1 \
  --local-files-only \
  --device cuda
```

The model runs on the host GPU. Tool commands and both public and hidden verification run in the
same network-disabled, non-root Docker boundary used by agent evaluation. Hidden source and output
are never placed in rollout records.

## Read the claim narrowly

The five MiniSWE tasks are deliberately tiny. A successful reference gate proves the exact-token
stateful optimization contract on the pinned model, suite, adapter, Docker image, and GPU. It does
not establish broad coding ability, long-horizon credit assignment, safe unrestricted tools, or
scalable rollout throughput.
