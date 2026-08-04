# NanoPT: an executable post-training course

NanoPT shows how a small base language model becomes an assistant, a reasoner, and a tool-using
agent. Equations lead to tested tensor code; tensor code leads to single-GPU experiments; every
reported claim leads back to inspectable artifacts.

## What you can learn here

- causal token coordinates, masks, log probabilities, entropy, and KL;
- completion-only LoRA SFT with explicit accumulation and checkpoint boundaries;
- controlled preference construction and white-box DPO;
- exact-token grouped rollouts and synchronous GRPO/RLVR;
- resettable agent environments, structured tools, budgets, and hidden verification;
- replay-linked multi-turn Agent SFT and context-policy experiments.

Start with the [prerequisites](getting-started/prerequisites.md), then use the
[21-chapter course map](course/index.md) to pair each chapter with its executable lab.

## Current release

NanoPT v0.2.0 adds exact-token Agent SFT. Its clean reference run retained 10/10 replayed source
trajectories, improved held-out action-token accuracy from 76.5% to 95.0%, produced 100% valid
actions under the trained full-transcript policy, and stayed within 13.94 GiB reserved VRAM.

Read the [v0.2 Agent SFT report](reference/v0.2-agent-sft-report.md) for the measurements and their
limits. The five-task suite is educational; its one held-out success is not a broad coding benchmark.

## Evidence vocabulary

- **Required** is part of a release contract.
- **Proposed** is an explicit starting point that still needs calibration.
- **Validated** means a reviewed evidence bundle passed the stated reference protocol.

The RTX 4070 Ti SUPER profile is validated for the pinned math pipeline and Agent SFT slice. CPU
labs validate equations and invariants, not GPU support. See the [artifact contract](reference/artifacts.md)
and [troubleshooting guide](troubleshooting.md) when reproducing a run.
