# Changelog

All notable user-facing changes are recorded here. NanoPT uses semantic versioning for the package
and course release, while the Python API remains explicitly experimental until v1.0.

## 0.1.0 — 2026-08-03

### Added

- A readable Base → LoRA SFT → DPO → synchronous GRPO/RLVR reference pipeline.
- Exact-token rollouts, deterministic synthetic arithmetic, strict parsing, protected evaluation,
  regression reports, and hash-linked resumable artifacts.
- A five-task MiniSWE environment with structured tools, isolated hidden verification,
  deterministic replay, and a hardened Docker reference backend.
- One prerequisite chapter, 20 numbered chapters, 20 executable local labs, reading guides,
  troubleshooting, systems simulations, a glossary, and contribution paths.
- Local, curriculum, reference-GPU, sandbox-security, and release validation gates that do not use
  GitHub Actions.

### Validated scope

- The pinned Qwen3 0.6B Base pipeline is validated on one NVIDIA RTX 4070 Ti SUPER 16 GB profile.
- MiniSWE evaluates agent policies in v0.1; it does not train an agent policy.
- Distributed training, accelerated rollout servers, QLoRA, Agent SFT, and Agent RL remain outside
  the required v0.1 path.
