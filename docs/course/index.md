# Course map: from tokens to agent systems

## Learning objectives

After reading this map, you should be able to:

- locate every NanoPT concept, implementation, lab, and retained reference result;
- distinguish CPU teaching evidence from GPU smoke and reference evidence;
- choose a reading path without confusing the milestone plan with the course itself;
- trace the Base → SFT → DPO → GRPO lineage before changing a checkpoint.

## How to use the course

Start with the [prerequisites](../getting-started/prerequisites.md). For each chapter, read the linked
implementation only after doing the numerical example. Run the CPU lab before the reference command;
the small lab teaches the invariant, while the reference run measures the complete pinned path.

| Chapter | Topic | Main lesson | Lab/evidence |
| ---: | --- | --- | --- |
| 0 | Course map and lineage | This page | CPU: `labs/18_artifact_lineage.py` |
| 1 | [Tokens and log probabilities](../foundations/tokens-masks-logprobs.md) | Causal coordinates and masks | CPU: labs 01–02 |
| 2 | [Optimization and LoRA](../foundations/optimization-and-lora.md) | Explicit update scope | CPU: lab 19 |
| 3 | [Evaluation before training](../evaluation/exact-generation-and-reports.md) | Parse, correctness, and samples first | CPU: lab 06; M3 reference |
| 4 | [Supervised fine-tuning](../sft/completion-only-training.md) | Completion-only LoRA | CPU: lab 07; M4 reference |
| 5 | [Preference data](../preferences/controlled-preferences.md) | Controlled chosen/rejected pairs | CPU: lab 08 |
| 6 | [Reward models](../preferences/reward-models.md) | Pairwise scalar ranking | CPU: lab 11 |
| 7 | [DPO](../preferences/dpo-training.md) | Policy/reference margins | CPU: lab 03; M5 reference |
| 8 | [RL foundations](../rl/reinforcement-learning-foundations.md) | State, action, return, on-policy data | CPU examples |
| 9 | [REINFORCE](../rl/reinforce.md) | Score-function gradient | CPU: lab 12 |
| 10 | [PPO](../rl/ppo.md) | Ratios and clipped local updates | CPU: lab 13 |
| 11 | [RLVR and GRPO](../grpo-rlvr/synchronous-grpo.md) | Grouped exact-token updates | CPU: labs 04/09; M6 reference |
| 12 | [Reward hacking](../grpo-rlvr/reward-hacking.md) | Parser/verifier attack surface | CPU: lab 14 |
| 13 | [From single-turn to agents](../agents/from-tool-call-to-trajectory.md) | Stateful tool trajectories | CPU: lab 10; M8 reference |
| 14 | [Verifiable agent tasks](../agents/task-authoring-and-verification.md) | Reset and hidden verification | CPU: lab 17; M8 reference |
| 15 | [Agent SFT](../agents/agent-sft.md) | Exact multi-turn action masks and replay lineage | CPU: lab 20; v0.2 reference |
| 16 | [Mini Agent RL](../agents/agent-rl.md) | Hidden outcomes, policy age, and multi-turn credit | CPU: lab 21; v0.3 reference |
| 17 | [Rollout infrastructure](../systems/rollout-infrastructure.md) | Long tails, partial work, staleness | Systems: lab 15 |
| 18 | [Production flywheels](../systems/production-flywheels.md) | Privacy and fixed-eval boundaries | Systems: lab 16 |
| 19 | [Reading modern systems](../reading/post-training-systems.md) | Compare reports without filling gaps | Reading guide |
| 20 | [End-to-end pipeline](../pipeline/end-to-end-recipe.md) | Hash-linked resumable stages | M7 reference |
| 21 | [Extending NanoPT](../extending/contribution-paths.md) | Add one auditable vertical slice | Contributor exercise |
| 22 | [RL from a systems perspective](../tutorials/rl-from-systems-perspective.md) | Experience, resume state, weights, cache, and admission | Systems: lab 22; v0.4 |

## Evidence tiers

- **CPU** labs run without a model download or GPU and are executed by the M9 curriculum gate.
- **Systems simulation** labs replace cluster mechanisms with deterministic counters; they do not
  claim cluster throughput.
- **GPU smoke** commands exercise a small real-model path but are non-representative.
- **Reference** commands were executed on the RTX 4070 Ti SUPER and are bound to compact evidence
  under `docs/reference/evidence/`.
- **Stretch** work is optional and cannot expand v0.1 support claims.

## Inspect lineage first

Run:

```bash
uv run python labs/18_artifact_lineage.py
```

It reads the committed M7 evidence without loading a model. Then compare the Base, SFT, DPO, and
GRPO checkpoint metrics in the [M7 report](../reference/m7-completion-report.md). The report is a
result of artifacts, not a substitute for them.

## Exercises

1. Pick one metric in the M7 evidence and trace it to the chapter that defines its semantics.
2. Explain why a CPU lab can validate an equation but cannot validate a 16 GB hardware claim.
3. Name the artifact that prevents an adapter from silently changing between pipeline stages.
