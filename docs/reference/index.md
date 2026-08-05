# Reference Library

The course navigation follows the recommended learning sequence. This library collects operational
references, validation reports, design decisions, and engineering specifications without placing
them in the primary learner path.

Use the course chapters to learn a concept. Use this page when you need an exact command, artifact
contract, measured result, or design rationale.

## Everyday references

- [CLI reference](cli.md): implemented commands and their behavioral contracts.
- [Run artifact contract](artifacts.md): files, hashes, lineage, and resumability guarantees.
- [Troubleshooting](../troubleshooting.md): common environment, data, and training failures.
- [Glossary](../glossary.md): project terminology and mathematical notation.
- [Acceptance criteria](../13_ACCEPTANCE_CRITERIA.md): the checked project-level requirements.

## Data, dependencies, and research

- [Synthetic arithmetic dataset card](../data/dataset-card.md)
- [Dependency and model license audit](dependency-license-audit.md)
- [Research and references](../14_RESEARCH_AND_REFERENCES.md)

## Validation reports

These reports bind public claims to commands, environments, artifacts, and known limitations.

??? abstract "Foundation through source-release validation"

    - [Repository foundation](m1-completion-report.md)
    - [Core mathematics and tasks](m2-completion-report.md)
    - [Base-model GPU path](m3-completion-report.md)
    - [Supervised fine-tuning GPU path](m4-completion-report.md)
    - [Preference optimization GPU path](m5-completion-report.md)
    - [GRPO and RLVR GPU path](m6-completion-report.md)
    - [End-to-end pipeline GPU path](m7-completion-report.md)
    - [Agent environment and Docker path](m8-completion-report.md)
    - [Curriculum validation](m9-completion-report.md)
    - [Source release audit](m10-completion-report.md)

??? abstract "Agent-training validation"

    - [Agent SFT validation](v0.2-agent-sft-report.md)
    - [Mini Agent RL validation](v0.3-agent-rl-report.md)

??? abstract "Release notes"

    - [v0.1.0](v0.1-release-notes.md)
    - [v0.2.0 — Agent SFT](v0.2-release-notes.md)
    - [v0.3.0 — Mini Agent RL](v0.3-release-notes.md)

## Architecture decisions

The ADRs explain why a boundary exists and which alternatives were rejected.

??? abstract "Architecture decision records"

    - [Project identity](../adr/0000-project-identity.md)
    - [Foundation boundaries](../adr/0001-foundation-boundaries.md)
    - [Run artifact writes](../adr/0002-run-artifact-writes.md)
    - [Local validation](../adr/0003-local-validation.md)
    - [Reference PyTorch wheel](../adr/0004-reference-pytorch-wheel.md)
    - [SFT schedule and checkpoints](../adr/0005-sft-schedule-and-checkpoints.md)
    - [Preferences and reference cache](../adr/0006-controlled-preferences-and-reference-cache.md)
    - [Synchronous exact-token GRPO](../adr/0007-synchronous-exact-token-grpo.md)
    - [Resumable hash-linked pipeline](../adr/0008-resumable-hash-linked-pipeline.md)
    - [Structured MiniSWE sandbox](../adr/0009-structured-mini-swe-sandbox.md)
    - [Exact-token Agent SFT](../adr/0010-exact-token-agent-sft.md)
    - [Fresh exact-token Agent GRPO](../adr/0011-fresh-exact-token-agent-grpo.md)
    - [Resumable rollout control plane](../adr/0012-resumable-rollout-control-plane.md)

## Engineering specifications

These documents describe the complete project contract. They are useful to contributors and
reviewers, but they are not required reading for the course.

??? abstract "Project and subsystem specifications"

    - [Vision and scope](../00_VISION_AND_SCOPE.md)
    - [Product requirements](../01_PRODUCT_REQUIREMENTS.md)
    - [Technical architecture](../02_TECHNICAL_ARCHITECTURE.md)
    - [Repository blueprint](../03_REPOSITORY_BLUEPRINT.md)
    - [Configuration and CLI](../04_CONFIGURATION_AND_CLI.md)
    - [Algorithm specifications](../05_ALGORITHM_SPECIFICATIONS.md)
    - [Data and tasks](../06_DATA_AND_TASKS.md)
    - [Agent environment](../07_AGENT_ENVIRONMENT.md)
    - [Hardware and performance](../08_HARDWARE_AND_PERFORMANCE.md)
    - [Evaluation and reporting](../09_EVALUATION_AND_REPORTING.md)
    - [Testing and security](../10_TESTING_CI_AND_SECURITY.md)
    - [Architectural decisions](../16_ARCHITECTURAL_DECISIONS.md)
    - [Risks and open questions](../17_RISKS_AND_OPEN_QUESTIONS.md)
