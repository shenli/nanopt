# Course map: from tokens to agent systems

## Learning objectives

After reading this map, you should be able to:

- locate every NanoPT concept, implementation, lab, and retained reference result;
- distinguish local teaching exercises from measured GPU and Docker evidence;
- choose a reading path without needing to understand the project's development milestones;
- trace the Base → SFT → DPO → GRPO lineage before changing a checkpoint.

## How to use the course

Start with [Getting Started](../getting-started/index.md). For each chapter, read the linked
implementation only after doing the numerical example. Run the local lab before the validation
command; the small lab teaches the invariant, while the linked evidence measures the complete
pinned path.

| Chapter | Topic | Main lesson | Practice and validation |
| ---: | --- | --- | --- |
| 0 | Course map and lineage | This page | Local CPU · lab 18 |
| 1 | [Tokens and log probabilities](../foundations/tokens-masks-logprobs.md) | Causal coordinates and masks | Local CPU · labs 01–02 |
| 2 | [Optimization and LoRA](../foundations/optimization-and-lora.md) | Explicit update scope | Local CPU · lab 19 |
| 3 | [Evaluation before training](../evaluation/exact-generation-and-reports.md) | Parse, correctness, and samples first | Local CPU · lab 06 · [GPU validation](../reference/m3-completion-report.md) |
| 4 | [Supervised fine-tuning](../sft/completion-only-training.md) | Completion-only LoRA | Local CPU · lab 07 · [GPU validation](../reference/m4-completion-report.md) |
| 5 | [Preference data](../preferences/controlled-preferences.md) | Controlled chosen/rejected pairs | Local CPU · lab 08 |
| 6 | [Reward models](../preferences/reward-models.md) | Pairwise scalar ranking | Local CPU · lab 11 |
| 7 | [DPO](../preferences/dpo-training.md) | Policy/reference margins | Local CPU · lab 03 · [GPU validation](../reference/m5-completion-report.md) |
| 8 | [RL foundations](../rl/reinforcement-learning-foundations.md) | State, action, return, on-policy data | Local CPU examples |
| 9 | [REINFORCE](../rl/reinforce.md) | Score-function gradient | Local CPU · lab 12 |
| 10 | [PPO](../rl/ppo.md) | Ratios and clipped local updates | Local CPU · lab 13 |
| 11 | [RLVR and GRPO](../grpo-rlvr/synchronous-grpo.md) | Grouped exact-token updates | Local CPU · labs 04/09 · [GPU validation](../reference/m6-completion-report.md) |
| 12 | [Reward hacking](../grpo-rlvr/reward-hacking.md) | Parser/verifier attack surface | Local CPU · lab 14 |
| 13 | [From single-turn to agents](../agents/from-tool-call-to-trajectory.md) | Stateful tool trajectories | Local CPU · lab 10 · [Docker validation](../reference/m8-completion-report.md) |
| 14 | [Verifiable agent tasks](../agents/task-authoring-and-verification.md) | Reset and hidden verification | Local CPU · lab 17 · [Docker validation](../reference/m8-completion-report.md) |
| 15 | [Agent SFT](../agents/agent-sft.md) | Exact multi-turn action masks and replay lineage | Local CPU · lab 20 · [GPU + Docker validation](../reference/v0.2-agent-sft-report.md) |
| 16 | [Mini Agent RL](../agents/agent-rl.md) | Hidden outcomes, policy age, and multi-turn credit | Local CPU · lab 21 · [GPU + Docker validation](../reference/v0.3-agent-rl-report.md) |
| 17 | [Rollout infrastructure](../systems/rollout-infrastructure.md) | Long tails, partial work, staleness | Local systems simulation · lab 15 |
| 18 | [Production flywheels](../systems/production-flywheels.md) | Privacy and fixed-eval boundaries | Local systems simulation · lab 16 |
| 19 | [Reading modern systems](../reading/post-training-systems.md) | Compare reports without filling gaps | Reading guide |
| 20 | [End-to-end pipeline](../pipeline/end-to-end-recipe.md) | Hash-linked resumable stages | [End-to-end GPU validation](../reference/m7-completion-report.md) |
| 21 | [Extending NanoPT](../extending/contribution-paths.md) | Add one auditable vertical slice | Contributor exercise |
| 22 | [Reinforcement Learning from a Systems Perspective](../tutorials/rl-from-systems-perspective.md) | Experience, resume state, weights, cache, and admission | Local systems simulation · lab 22 |

## Evidence tiers

- **Local CPU** labs run without a model download or GPU and are checked by the local curriculum
  gate.
- **Systems simulation** labs replace cluster mechanisms with deterministic counters; they do not
  claim cluster throughput.
- **GPU smoke** commands exercise a small real-model path but are non-representative.
- **Validated GPU/Docker** commands were executed on the named reference environment and are bound to compact evidence
  under `docs/reference/evidence/`.
- **Stretch** work is optional and cannot expand published support claims.

## Inspect lineage first

Run:

```bash
uv run python labs/18_artifact_lineage.py
```

It reads the committed pipeline evidence without loading a model. Then compare the Base, SFT, DPO,
and GRPO checkpoint metrics in the [end-to-end validation report](../reference/m7-completion-report.md).
The report is a result of artifacts, not a substitute for them.

## Exercises

1. Pick one metric in the end-to-end evidence and trace it to the chapter that defines its semantics.
2. Explain why a CPU lab can validate an equation but cannot validate a 16 GB hardware claim.
3. Name the artifact that prevents an adapter from silently changing between pipeline stages.
