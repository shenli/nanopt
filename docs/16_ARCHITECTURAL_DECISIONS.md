# Architectural Decisions

These are proposed initial ADRs. The coding agent should create individual files under `docs/adr/`, preserve decision history, and update status only through review.

## ADR-001: Build an executable course, not a production framework

**Status:** Proposed for acceptance.

**Decision:** Optimize for readable algorithms, labs, artifacts, and one validated path. Do not make distributed scalability or broad configurability a v0.1 goal.

**Reasoning:** A production abstraction can hide exactly the details the course is meant to teach.

**Consequence:** Some code duplication is accepted; future backends must not remove the reference loops.

## ADR-002: Use Qwen3 0.6B Base for the official path

**Status:** Proposed.

**Decision:** Start from `Qwen/Qwen3-0.6B-Base`; use the post-trained sibling only for isolated debugging comparisons.

**Reasoning:** It is small enough for consumer hardware, modern, publicly available, and has an official base/post-trained relationship.

**Consequence:** Renderer and architecture assumptions are isolated so a future model can be added.

## ADR-003: Validate BF16 LoRA before adding QLoRA

**Status:** Proposed.

**Decision:** Keep base weights frozen in BF16 and train LoRA adapters with standard AdamW.

**Reasoning:** At 0.6B, quantization is probably unnecessary and would add bitsandbytes/version/debugging complexity.

**Consequence:** 4-bit support is an optional later backend, not a v0.1 dependency.

## ADR-004: Own core SFT, DPO, and GRPO loops

**Status:** Proposed.

**Decision:** Implement mathematical objectives and training order in NanoPT. Use Transformers and PEFT as model infrastructure. Use TRL only for optional parity examples.

**Reasoning:** Learners need to see masks, log probabilities, margins, advantages, ratios, and update timing.

**Consequence:** NanoPT assumes responsibility for correctness tests and must track changes in upstream model APIs.

## ADR-005: Use synthetic exact-answer tasks for the golden path

**Status:** Proposed.

**Decision:** Generate arithmetic/symbolic tasks from safe ASTs with exact verifiers.

**Reasoning:** This controls data licensing, leakage, difficulty, reward, and adversarial cases.

**Consequence:** Results demonstrate mechanics in a narrow domain, not broad assistant quality.

## ADR-006: Precompute DPO reference log probabilities

**Status:** Proposed.

**Decision:** Cache reference chosen/rejected log probabilities from the SFT checkpoint, then continue a copied adapter as DPO policy.

**Reasoning:** Saves GPU memory and makes policy/reference semantics explicit.

**Consequence:** Cache fingerprinting and invalidation become release-critical.

## ADR-007: Use a simple synchronous sampler for GRPO

**Status:** Proposed.

**Decision:** Implement exact autoregressive generation in PyTorch/Transformers with no vLLM dependency. Use untruncated temperature-one sampling in the reference training recipe.

**Reasoning:** Correct behavior-policy log probabilities and token identity are easier to verify.

**Consequence:** Rollout throughput will be lower; acceleration is deferred until profiling.

## ADR-008: Keep GRPO objective variants explicit

**Status:** Proposed.

**Decision:** Name advantage scaling, clipping, KL, and loss normalization independently in config and reports.

**Reasoning:** “GRPO” implementations differ, and trainer defaults evolve.

**Consequence:** More configuration fields, but experiments remain interpretable.

## ADR-009: Compose hardware, model, experiment, and recipe configs

**Status:** Proposed.

**Decision:** Use small typed YAML profiles with strict deterministic resolution and provenance.

**Reasoning:** Hardware compatibility should not be hard-coded into algorithms.

**Consequence:** The resolver is a core subsystem and must reject implicit/unknown behavior.

## ADR-010: Use Docker with structured tools for agent tasks

**Status:** Proposed.

**Decision:** No arbitrary shell. Model actions are typed tool calls executed in a no-network, non-root, resource-limited Docker sandbox; hidden verification occurs separately.

**Reasoning:** A useful agent lab needs stateful execution, but model output is untrusted.

**Consequence:** Linux and Docker are required for the reference agent lab; the security threat model must remain explicit.

## ADR-011: Agent environment before Agent RL

**Status:** Proposed.

**Decision:** v0.1 builds and evaluates the environment; v0.2 adds Agent SFT; v0.3 adds Agent RL.

**Reasoning:** Training against a nondeterministic or insecure environment produces uninterpretable results.

**Consequence:** The first release teaches the full environment contract without claiming optimized agent capability.

## ADR-012: Generate local reports and keep hosted telemetry optional

**Status:** Proposed.

**Decision:** JSONL artifacts plus Markdown/HTML reports are required; W&B or other services may be optional integrations later.

**Reasoning:** The course should work offline after artifacts are cached and should remain auditable without an account.

## ADR-013: English-only public repository

**Status:** Accepted by project owner.

**Decision:** Code, comments, docs, reports, examples, and project governance are English.

**Consequence:** Background source materials may be multilingual in the private handoff, but must be synthesized into original English public documentation.

## ADR-014: Evidence-backed hardware profiles

**Status:** Accepted by project owner.

**Decision:** Start with one 4070 Ti SUPER profile and add other consumer GPUs through measured evidence.

**Consequence:** Same-VRAM devices are not automatically supported, and initial profile remains unvalidated until full runs pass.
