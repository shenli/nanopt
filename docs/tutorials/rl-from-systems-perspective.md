# RL from a systems perspective: trace one experience end to end

Reinforcement learning is often introduced as a sequence of algorithms: Bellman equations,
REINFORCE, actor-critic, PPO, and GRPO. That sequence explains how feedback can become a gradient,
but it does not explain where the feedback comes from or whether the experience is safe to train
on. For an LLM agent, those are systems questions.

This tutorial follows one coding-agent experience from an immutable task snapshot to a training
admission decision. It connects NanoPT's implemented v0.3 Agent RL path to the deterministic v0.4
resumable-rollout laboratory.

## Learning objectives

After completing this tutorial, you should be able to:

- map policy, harness, environment, verifier, trainer, and artifact store onto an RL loop;
- distinguish environment state from the observation placed in the model context;
- explain why a tool action has both agent-level and token-level coordinates;
- name the model state and world state required to resume a partial rollout;
- explain why a prefix cache is identified by both tokens and policy weights;
- compare episode-boundary and action-boundary weight synchronization;
- decide whether a completed trajectory is fresh, stale, or mixed-policy;
- separate an executable control-plane simulation from a measured production runtime.

## 1. Start with the experience pipeline

A post-training system has two related pipelines:

```text
parameter pipeline
checkpoint → forward/backward → optimizer step → new checkpoint

experience pipeline
task → reset → observe → act → transition → verify → reward → admission
```

Supervised learning can begin with a fixed record containing an input and target. Online Agent RL
cannot. The current policy must be allowed to choose an action, the environment must execute it,
and an independent verifier must evaluate the resulting world.

NanoPT separates this loop into five responsibilities:

| Responsibility | NanoPT component | Question it answers |
| --- | --- | --- |
| Policy | Qwen + selected LoRA adapter | What token is sampled next? |
| Harness | Agent policy/context code | How does output become a structured action? |
| Environment | MiniSWE workspace and sandbox | What does the action change? |
| Verifier | Public tests and isolated hidden checks | Did the world reach the objective? |
| Trainer | Exact-token Agent RL update | How does accepted feedback change weights? |

The artifact layer crosses every row. A claim is reproducible only when a task snapshot, policy
version, exact action coordinates, state transition, reward, and update boundary can be related
after the process exits.

## 2. One task has more state than the model can see

Consider a tiny repository task: fix a function without mutating its input. The complete
environment state can include:

```text
immutable task snapshot
current workspace files
public-test results
remaining tool budget
elapsed-time budget
hidden verifier source
hidden verifier workspace
```

The model observation contains only the allowed projection:

```text
task description
visible conversation history
allow-listed tool schemas
previous public tool result
remaining visible budget
```

Therefore:

$$
o_t = h(s_t), \qquad o_t \neq s_t.
$$

The observation function $h$ is part of the product and training design. If hidden tests enter the
context, the reward boundary is broken. If an important public error is omitted, the policy is
making a decision from an unnecessarily impoverished observation.

NanoPT's hidden outcome is produced only after episode termination. It is written to training
artifacts but never appended to the next model observation.

## 3. One tool action has two coordinate systems

At the agent level, the policy may choose:

```json
{"tool":"run_tests","arguments":{}}
```

At the model level, that action is a sequence of sampled token IDs. The environment waits for a
complete structured action before transitioning, while the gradient is computed at causal token
coordinates.

For action turn $j$, v0.3 retains:

```text
exact prompt token IDs
exact sampled token IDs
aligned current-action mask
FP32 behavior log probabilities
frozen Agent SFT reference log probabilities
parse status and tool identity
behavior-policy version
task and snapshot identity
```

The first sampled action token is scored by causal prediction coordinate $P-1$, where $P$ is the
prompt length. Decoding the JSON and tokenizing it again would create a different training record;
NanoPT consumes the stored online IDs directly.

The v0.4 simulation uses obviously synthetic IDs because it does not load a model. The record
shape is still useful to systems engineers: it shows which data must cross the rollout/trainer
boundary without claiming the IDs came from Qwen.

## 4. Reset before comparing outcomes

Suppose four episodes share one task. Their workspaces must be independent copies of the same
immutable snapshot:

```text
snapshot H
├── reset → episode 0 workspace
├── reset → episode 1 workspace
├── reset → episode 2 workspace
└── reset → episode 3 workspace
```

If episode 1 inherits episode 0's edit, reward differences no longer identify policy behavior.
The group-relative advantage would be contaminated by different starting states.

For terminal rewards $R_1,\ldots,R_G$, NanoPT v0.3 uses:

$$
A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}.
$$

The task ID is insufficient evidence of identical starts. A rollout group also records the
immutable snapshot hash and rejects mixed values.

## 5. Verification and credit assignment are different problems

An independent hidden verifier can say that the final workspace earned reward 1.0. It cannot prove
which earlier read, edit, or test action caused success.

NanoPT v0.3 assigns the episode advantage to every sampled action token. Its retained
counterfactual asks how much coverage would remain if only the terminal action received credit:

```text
all-action credit:       3,649 sampled tokens
terminal-action credit:    340 sampled tokens
terminal fraction:         9.32%
```

This is a coverage study, not a separately trained policy. It makes the modeling choice visible:
terminal-only credit cannot reinforce earlier inspection or editing, while all-action credit may
also reinforce unnecessary steps in a successful episode.

## 6. Why partial rollout changes the data contract

Short synchronous episodes can finish before one policy update. Long agent episodes create a
straggler problem:

```text
short episodes finish
→ trainer publishes policy v1
→ long episode is still running under v0
→ more short episodes finish
→ trainer publishes v2 and v3
```

The runtime has three broad choices:

1. wait for the long episode and leave resources idle;
2. discard and restart partial work under the new policy;
3. pause the episode, update, and resume it with an explicit synchronization rule.

The original scheduler lab compares the first two trade-offs abstractly. The v0.4 lab implements
the third as a deterministic state-and-cache simulation.

## 7. A resumable checkpoint has two halves

Saving conversation text is not enough to resume an agent. NanoPT v0.4 pairs model execution state
with world execution state.

```text
PartialRolloutCheckpoint
├── model state
│   ├── exact prompt token IDs
│   ├── sampling RNG counter
│   ├── safe stop state: between_actions
│   ├── behavior-policy version and hash
│   ├── generation-config hash
│   └── prefix-cache key
└── world state
    ├── immutable snapshot hash
    ├── current workspace hash
    ├── event cursor
    └── remaining tool budget
```

The checkpoint payload is hash-bound. Resume rejects it when:

- the task snapshot changed;
- the model RNG counter and world event cursor disagree;
- the workspace hash does not derive from that snapshot and cursor;
- the remaining budget is invalid;
- any payload field changed without a new payload hash.

The lab pauses only between complete actions. Pausing in the middle of a JSON tool call would also
require tokenizer/parser state and would expose the environment to an incomplete action.

## 8. Weight synchronization has no free option

When the trainer publishes policy v1 while an episode is paused, the worker can follow either of
two explicit rules.

### Keep weights until the episode boundary

The worker resumes with v0 and finishes the entire episode under one behavior policy.

Benefits:

- the episode has one policy identity;
- a v0 prefix cache remains numerically reusable;
- action probabilities are internally consistent with the stored v0 behavior policy.

Cost:

- the episode can be several policy versions stale when it terminates;
- the runtime must retain or reload old weights;
- a strict fresh trainer rejects it.

### Synchronize at an action boundary

The worker finishes the current action, checkpoints, then resumes the next action with v1.

Benefits:

- later actions come from newer weights;
- no single structured action is split across policy versions.

Cost:

- the complete episode contains multiple behavior-policy versions;
- an old-policy KV cache is invalid under the new weights;
- the objective needs per-action or per-segment off-policy semantics that NanoPT does not claim.

Updating weights in the middle of an action and retaining only the final version is not a third
valid rule. It destroys the behavior identity required by probability-ratio objectives.

## 9. Cache identity includes the policy

A prefix cache is a function of both token IDs and model parameters:

$$
K = f_{\theta}(x_{1:P}).
$$

The same tokens evaluated by policy v0 and policy v1 generally produce different keys and values.
The v0.4 cache key therefore hashes:

```text
(behavior policy hash, exact prompt token IDs)
```

In the reference simulation, keeping episode weights restores three cached prefixes with no
recomputation. Synchronizing at action boundaries changes the policy hash, causes three misses,
and recomputes 42 synthetic prefix tokens. These are deterministic teaching counts, not measured
GPU latency or memory traffic.

## 10. Admission is separate from collection

A trajectory can be useful evidence without being safe training data. For trainer policy version
$v$, NanoPT defines action lag as:

$$
\ell_j = v - v_j^{\text{behavior}}.
$$

The v0.4 admission record reports:

- every action's behavior-policy version;
- whether the episode mixes versions;
- maximum policy lag;
- how many actions satisfy a bounded-lag counterfactual;
- the reason the complete episode is accepted or rejected.

The strict rule admits a complete episode only when it has one behavior policy and satisfies the
configured lag limit. The reference limit is zero. A bounded-action count is reported for study,
but no simulated action is passed to a model update.

This separation prevents a common systems failure: collecting a rich execution log and assuming
that its existence makes it valid PPO or GRPO input.

## 11. Run the v0.4 laboratory

The smallest lesson prints the comparison directly:

```bash
uv run python labs/22_resumable_rollouts.py
```

Expected result:

```text
mode              mixed  stale  cache hit/miss  recomputed prompt tokens
episode_boundary      0      1     3/0                           0
action_boundary       1      1     0/3                          42
Resumable-rollout systems lab passed.
```

To retain a complete artifact bundle:

```bash
uv run nanopt systems simulate \
  --experiment resumable_rollouts \
  --run-id systems-tutorial
```

Inspect:

```text
artifacts/runs/systems-tutorial/
├── actions.jsonl
├── partial_checkpoints.jsonl
├── weight_sync_events.jsonl
├── admission_decisions.jsonl
├── summary.json
├── report.md
└── run_manifest.json
```

Start with `summary.json`, then find `trajectory-1` in the admission decisions. Under
`episode_boundary`, all eight actions use v0 and finish three versions stale. Under
`action_boundary`, the versions are `[0, 0, 1, 1, 2, 2, 3, 3]`; the two newest actions are fresh,
but the episode is not a single-policy GRPO sample.

## 12. Read the implementation in control-flow order

The systems slice is intentionally small:

1. [`resumable_rollouts.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/systems/resumable_rollouts.py)
   defines checkpoint, cache, sync, and admission contracts.
2. [`run.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/systems/run.py)
   executes both strategies and writes the artifact bundle.
3. [`test_resumable_rollouts.py`](https://github.com/shenli/nanopt/blob/main/tests/unit/systems/test_resumable_rollouts.py)
   contains hand-checkable trade-off and tamper tests.
4. [`22_resumable_rollouts.py`](https://github.com/shenli/nanopt/blob/main/labs/22_resumable_rollouts.py)
   is the minimal executable lesson.

The earlier [Mini Agent RL chapter](../agents/agent-rl.md) explains the real exact-token update.
The [rollout infrastructure chapter](../systems/rollout-infrastructure.md) explains the scheduling
background, and the [production flywheel chapter](../systems/production-flywheels.md) explains how
real failures become authorized reproducible tasks.

## 13. Implemented, simulated, and omitted

| Mechanism | Evidence tier |
| --- | --- |
| Exact online Agent RL IDs/logprobs and fresh update | Implemented and GPU validated in v0.3 |
| Docker task reset, tools, public tests, hidden verification | Implemented and reference validated |
| Model/world checkpoint contract and tamper checks | Implemented as deterministic CPU records |
| Partial pause/resume scheduling | Deterministic CPU simulation |
| External prefix-cache hit/miss/eviction accounting | Deterministic CPU metadata simulation |
| Weight publication and admission decisions | Deterministic CPU simulation |
| Real KV allocation, transfer, or measured throughput | Not implemented |
| vLLM/SGLang integration or distributed workers | Not installed or claimed |
| Mid-generation process recovery | Not implemented |
| Off-policy training from mixed/stale episodes | Measured/classified, never performed |
| Firecracker pause/resume/fork/snapshot | Industrial reading only |

The distinction is part of the result. A simulation can validate state-machine invariants; it
cannot validate throughput, fault tolerance, numerical parity, or hardware capacity.

## 14. Common mistakes

- Treating context text as the complete environment state.
- Saving a sandbox but not the model RNG, policy version, or prompt IDs.
- Saving prompt IDs but not the workspace and remaining budget.
- Reusing KV state after changing model weights.
- Recording one policy version for an episode that crossed several versions.
- Calling action-boundary synchronization on-policy because the last action is fresh.
- Assuming PPO clipping makes arbitrarily stale data safe.
- Using a public test result as both a model observation and a hidden evaluation claim.
- Reporting simulated cache tokens as GPU throughput.
- Letting collected data enter training before a separate admission decision.

## 15. Industrial mapping and primary reading

[Kimi k1.5](https://arxiv.org/abs/2501.12599) motivates partial rollouts for long-tail reasoning.
[Kimi K3](https://arxiv.org/abs/2607.24653) describes million-token agentic RL with partial
rollouts, an external CPU KV-cache pool, adaptive throttling, and persistent sandbox state.
[HybridFlow/veRL](https://arxiv.org/abs/2409.19256) presents distributed dataflow and placement
choices for policy, reference, reward, rollout, and training workers.

NanoPT extracts small invariants from those systems. It does not reproduce their models, clusters,
cache engines, sandboxes, task distributions, or performance results.

## Exercises

1. Set `freshness.max_policy_lag=1`. Which actions in the long mixed-policy episode become bounded-
   lag eligible, and why is the complete episode still rejected?
2. Set `cache.capacity_entries=0`. Predict hit, miss, and recomputation counts before running it.
3. Add a third worker and two long episodes. How does a one-entry external cache change eviction?
4. Design a checkpoint for pausing in the middle of token generation. List every additional state
   field and explain why NanoPT currently chooses an action boundary instead.
5. Propose an off-policy objective for mixed episodes. State what additional assumptions and tests
   would be required before setting `used_for_model_update` to true.
