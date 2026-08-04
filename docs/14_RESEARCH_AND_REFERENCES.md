# Research and References

This is the reading and implementation reference map for NanoPT. Prefer primary papers, official technical reports, official model cards, and official repositories. URLs and APIs can change; the repository should record the versions actually used in each release.

## 1. Repository-design references

### nanoGPT

- Repository: https://github.com/karpathy/nanoGPT
- Role for NanoPT: minimal, hackable code and a short path from command to training loop.
- Caveat: nanoGPT itself now points users toward nanochat and should be treated as a design inspiration, not a current post-training implementation.

### nanochat

- Repository: https://github.com/karpathy/nanochat
- Role for NanoPT: an end-to-end experimental harness with readable stages and a clear “not a giant framework” philosophy.
- Difference: NanoPT starts from an existing base model and makes post-training algorithms and environments the primary subject.

### CleanRL

- Repository: https://github.com/vwxyzjn/cleanrl
- Role for NanoPT: one complete algorithm per readable file, with limited abstraction.

## 2. Reference model and efficient adaptation

### Qwen3 0.6B Base

- Model card: https://huggingface.co/Qwen/Qwen3-0.6B-Base
- Technical report: https://arxiv.org/abs/2505.09388
- Role: official base checkpoint for the v0.1 learning path.
- Important implementation note: the model card requires a Transformers version with Qwen3 support; record the exact resolved version and model revision.
- License verified for the pinned v0.1 revision on 2026-08-03: Apache-2.0. NanoPT references the
  model and does not redistribute its weights.

### Qwen3 0.6B post-trained sibling

- Model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Role: optional debugging baseline for tool calling or environment plumbing; not the starting checkpoint for the official learning pipeline.

### LoRA

- Paper: https://arxiv.org/abs/2106.09685
- PEFT documentation: https://huggingface.co/docs/peft/
- Role: memory-efficient adapter training on the consumer-GPU reference path.

### QLoRA

- Paper: https://arxiv.org/abs/2305.14314
- Role: optional later memory backend. NanoPT v0.1 first validates BF16 LoRA on 0.6B rather than adding quantization complexity prematurely.

## 3. Full post-training pipelines

### Tülu 3

- Paper: https://arxiv.org/abs/2411.15124
- Code: https://github.com/allenai/open-instruct
- Reproduction guide: https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md
- Role: the clearest open reference for a staged SFT → preference optimization → RLVR pipeline, data mixtures, evaluations, and regressions.

### The Llama 3 Herd of Models

- Paper: https://arxiv.org/abs/2407.21783
- Role: large-scale assistant post-training, synthetic data, preference data, rejection sampling, capability and safety tuning.
- Caveat: many production details are summarized rather than fully reproducible.

## 4. Preference optimization

### Direct Preference Optimization

- Paper: https://arxiv.org/abs/2305.18290
- Role: standard offline chosen/rejected objective and the policy/reference margin implemented in NanoPT.

### Hugging Face TRL

- Repository: https://github.com/huggingface/trl
- Documentation: https://huggingface.co/docs/trl/
- Role: optional parity and practical-library examples after the white-box code works.
- Version warning: trainer defaults and even named objectives evolve. NanoPT must compare exact masks, reductions, reference behavior, and loss variants rather than assuming name-level equivalence.

## 5. Policy optimization and reasoning RL

### Proximal Policy Optimization

- Paper: https://arxiv.org/abs/1707.06347
- Role: probability ratios, clipped updates, actor-critic foundations, and minibatch reuse.
- NanoPT scope: a tiny teaching implementation in v0.1; no full 0.6B PPO/RLHF reference pipeline.

### DeepSeekMath

- Paper: https://arxiv.org/abs/2402.03300
- Role: original GRPO source and grouped, value-model-free policy optimization for mathematical reasoning.

### DeepSeek-R1

- Paper: https://arxiv.org/abs/2501.12948
- Role: reasoning RL, cold-start data, pure-RL behavior, rejection sampling, general SFT, and later-stage RL.

### Kimi k1.5

- Paper: https://arxiv.org/abs/2501.12599
- Role: long-context reasoning RL, partial rollout, prompt sampling, length control, and scaling behavior.

## 6. Scalable RL systems

### HybridFlow and veRL

- Paper: https://arxiv.org/abs/2409.19256
- Repository: https://github.com/volcengine/verl
- Agent-loop docs: https://github.com/volcengine/verl/tree/main/docs
- Role: mapping actor, critic, reference, reward, rollout, devices, parameter synchronization, and multi-turn agent loops.
- NanoPT scope: system-reading target and later optional backend, not a v0.1 dependency.

### OpenRLHF

- Repository: https://github.com/OpenRLHF/OpenRLHF
- Role: Ray/vLLM/DeepSpeed-based scalable RLHF and Agentic RL reference.
- NanoPT scope: architecture comparison only in the initial release.

### Gymnasium

- Documentation: https://gymnasium.farama.org/
- Role: conventional environment terminology—observation, action, reset, step, reward, termination, and truncation.

## 7. Agent RL and long-horizon infrastructure

### Kimi K3 Technical Report

- Official paper: https://arxiv.org/abs/2607.24653
- Official repository: https://github.com/MoonshotAI/Kimi-K3
- Relevant sections in the official report:
  - post-training method;
  - RL task synthesis and agentic environments;
  - partial rollout and stale trajectories;
  - reasoning-effort and verbosity budgets;
  - configurable white-box harnesses;
  - deterministic/hidden/generative verification;
  - long-context rollout infrastructure;
  - external KV-cache retention;
  - pause/resume/fork/snapshot sandbox lifecycle.
- Role for NanoPT: industrial-scale contrast for the small agent environment and systems chapters.
- Redistribution warning: link to the official paper or repository rather than committing a local
  PDF copy.

## 8. Suggested reading order for contributors

1. NanoPT vision and algorithm specs.
2. nanochat and CleanRL for code style.
3. Qwen3 model card and PEFT LoRA docs.
4. DPO paper.
5. PPO paper, focusing on ratio and clipping.
6. DeepSeekMath GRPO section.
7. Tülu 3 and Open Instruct implementation.
8. DeepSeek-R1 and Kimi k1.5.
9. HybridFlow/veRL and OpenRLHF.
10. Kimi K3 report sections on environment and infrastructure.

## 9. Fixed report-comparison template

For each major post-training report, document:

- base model;
- stage sequence;
- SFT/cold-start data;
- preference stage;
- RL algorithm and whether a value model is used;
- rollout granularity;
- reward types and verifiers;
- KL, length, and reasoning-budget controls;
- task synthesis;
- tool/agent environment;
- reward-hacking defenses;
- evaluation and regressions;
- capability consolidation/distillation;
- rollout/training infrastructure;
- sandbox/state persistence;
- released code/data;
- explicitly undisclosed details.

Do not fill missing report details from intuition. Mark them unknown.
