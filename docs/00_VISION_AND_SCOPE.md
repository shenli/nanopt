# Vision and Scope

## One-sentence definition

**NanoPT is a minimal, executable course that shows how a base language model becomes an assistant, a reasoning model, and eventually an agent, with a reference path validated on one consumer GPU.**

## The problem

Modern post-training knowledge is fragmented across papers, technical reports, production frameworks, and small demonstrations. Learners commonly encounter one of four failure modes:

1. formulas without executable data flow;
2. trainer APIs that hide the key mathematics;
3. research recipes that assume multi-node GPU clusters;
4. toy reinforcement-learning examples that never connect to language-model post-training or agent environments.

NanoPT connects these layers. A learner should be able to move continuously from a hand-computable loss, to a tested tensor function, to a single-GPU run, to an industrial-system reading map.

## Product principles

### 1. White-box before framework

The project implements the core mechanics of SFT, DPO, and GRPO directly. Mature libraries are valuable, but they become useful educational references only after the learner can explain what they abstract away.

### 2. One complete path beats many shallow examples

The official path uses one model family, one task family, one verifier design, one evaluation harness, and one reference GPU. Alternate models and libraries are extensions, not the foundation.

### 3. Consumer-GPU support is evidence-based

The project begins with one hardware profile. New profiles are accepted only with reproducible manifests, peak-memory measurements, throughput measurements, and successful smoke and reference runs.

### 4. Algorithms and environments are separate systems

Reasoning RLVR can be studied with static prompts and deterministic answer verifiers. Agent RL additionally needs stateful environments, tools, state transitions, isolation, resets, budgets, and hidden verification. The repository makes this boundary explicit.

### 5. Evaluation is part of training

A higher training reward is not sufficient evidence of improvement. Every stage is evaluated on fixed held-out data, compositional generalization splits, output-format robustness, reward-hacking probes, length, entropy, and distance from prior checkpoints.

### 6. Educational code is allowed to repeat itself

A small amount of repetition is preferable to a deep hierarchy that obscures control flow. Reuse stable mathematical primitives, data schemas, evaluation, and artifact writing; keep algorithm entry points readable as vertical slices.

## Target audience

The primary learner:

- is comfortable reading Python;
- understands basic Transformer inference and next-token prediction;
- has seen introductory reinforcement learning but may not remember the formulas;
- wants to read modern post-training reports critically;
- has access to one consumer NVIDIA GPU;
- values engineering systems as much as algorithm names.

The course should explain probability, log probabilities, cross-entropy, gradients, KL divergence, policy gradients, baselines, advantages, clipping, on-policy/off-policy data, and verifier design from intuition through implementation.

## v0.1 scope

### Required

- project scaffolding, CLI, configuration, run artifacts, and documentation site;
- support profile for one RTX 4070 Ti SUPER 16 GB GPU;
- Qwen3 0.6B Base model integration;
- synthetic exact-answer arithmetic/reasoning data;
- baseline evaluation;
- completion-only LoRA SFT;
- deterministic construction of preference pairs;
- LoRA DPO with precomputed reference log probabilities;
- synchronous GRPO/RLVR with exact sampled token IDs;
- correctness and format rewards;
- checkpoint comparison and regression reporting;
- a resettable MiniSWE-style agent environment and trajectory recorder;
- CPU tests and opt-in GPU tests;
- an English course covering the implemented path and the missing industrial layers.

### Educational but not part of the full-model reference pipeline

- REINFORCE and PPO on a tiny model or tabular environment;
- a hand-built reward-model loss;
- simulations of partial rollout scheduling and stale trajectories;
- TRL parity examples.

### Explicitly out of scope for v0.1

- multi-GPU or multi-node training;
- production distributed RLHF;
- asynchronous optimization;
- online weight synchronization to inference workers;
- vLLM co-location;
- 7B-model GRPO on 16 GB;
- learned reward models in the reference pipeline;
- agent-policy optimization in the stateful coding environment;
- unrestricted shell tools;
- real email, calendar, browser, or production SaaS credentials;
- million-token trajectories;
- claims of reproducing frontier-model results.

## Version trajectory

- **v0.1 — Math post-training pipeline and agent environment.** SFT, DPO, GRPO, evaluation, and MiniSWE rollouts on the reference GPU.
- **v0.2 — Multi-turn trajectories and Agent SFT (complete).** Exact token/action serialization,
  replay-linked tool and recovery data, supervised agent behavior, and context-policy evaluation.
- **v0.3 — Mini Agent RL (complete).** Fresh grouped rollouts, exact-token short-horizon policy
  optimization, policy-age evidence, credit-assignment experiments, and tool-budget evaluation.
- **v0.4 — Systems laboratory.** Partial rollout, staleness, external state, and optional accelerated rollout integrations.
- **v1.0 — Stable executable course.** Validated profiles, polished curriculum, reproducible results, and contribution process.

## Naming

- Repository: `nanopt`
- Python import: `nanopt`
- CLI: `nanopt`
- Prose brand: `NanoPT`
- Full name: `Nano Post-Training`

Before publishing, verify repository, package-index, domain, and trademark availability. The name has unrelated uses outside language-model software, so the project should consistently pair NanoPT with “Nano Post-Training” in titles and metadata.
