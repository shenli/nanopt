# NanoPT

NanoPT is an executable course and white-box reference implementation for post-training a small
base language model. The project connects hand-computable objectives to tested tensor functions,
single-GPU experiments, inspectable artifacts, and a secure stateful agent environment.

Milestone 1 is the repository foundation. Milestone 2 provides the CPU-tested mathematical core,
model/adapter integration contracts, and deterministic synthetic data. Milestone 3 provides exact
generation, example-first evaluation, statistical summaries, shareable reports, and a completed
reference-GPU base evaluation. Milestone 4 adds completion-only LoRA SFT and protected generation
evidence. Milestones 5 and 6 add controlled DPO and exact-token GRPO. Milestone 7 joins the complete
path with hash-linked, independently resumable stages and validates the reference GPU. Start with
the [prerequisites](getting-started/prerequisites.md), then follow the course in navigation order.
Milestone 8 adds the five-task MiniSWE environment, structured tools, isolated verification,
deterministic replay, and a validated hardened-Docker reference path.

## Status vocabulary

- **Required** means the release contract requires the behavior.
- **Proposed** means the setting is an explicit starting point that still needs calibration.
- **Validated** means a reviewed evidence bundle passed the complete reference protocol.

The RTX 4070 Ti SUPER configuration is validated for the pinned Base → SFT → DPO → GRPO reference
path. The [M7 report](reference/m7-completion-report.md) states the exact scope and evidence.
The [M8 report](reference/m8-completion-report.md) documents the separate agent-environment scope,
including its unsuccessful but retained base-model baseline.

Use the [course map](course/index.md) to connect all 20 chapters to their executable labs and prior
reference evidence. The [troubleshooting guide](troubleshooting.md) records actual lessons from the
SFT, DPO, GRPO, pipeline, and Docker validation runs; the [glossary](glossary.md) fixes notation
before comparing external reports.
