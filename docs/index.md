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
- replay-linked multi-turn Agent SFT and context-policy experiments;
- fresh exact-token Mini Agent RL with policy-age, credit, and tool-budget studies;
- resumable-rollout state, policy synchronization, cache, and admission simulations.

Start with [Getting Started](getting-started/index.md), then use the
[23-chapter course map](course/index.md) to pair each chapter with its executable lab.

## Agent RL on one consumer GPU

NanoPT includes exact-token Mini Agent RL, validated on one consumer GPU with 16 GB VRAM—the pinned
course path does not require an H100 or B200. The clean run collected four non-degenerate groups,
16 episodes, and 80 action turns; sampled action validity was 91.25%, mean outcome reward was
0.6719, and peak reserved VRAM was 14.094 GiB. The selected post-update boundary retained validation
reward 1.0; the later terminal boundary scored 0.0 and remains disclosed.

Read the [Mini Agent RL validation report](reference/v0.3-agent-rl-report.md) for the measurements and
their limits. The tiny task suite validates this release contract, not broad coding ability.

The rollout-systems tutorial is executable locally. Read
[Reinforcement Learning from a Systems Perspective](tutorials/rl-from-systems-perspective.md), then run
`uv run python labs/22_resumable_rollouts.py`.

## Evidence vocabulary

- **Required** is part of a release contract.
- **Proposed** is an explicit starting point that still needs calibration.
- **Validated** means a reviewed evidence bundle passed the stated reference protocol.

The pinned math pipeline, Agent SFT, and Mini Agent RL slices are validated on a single consumer GPU
with 16 GB VRAM. This is a measured configuration, not a claim that every 16 GB GPU or software
stack is compatible. CPU labs validate equations and invariants, not GPU support. See the
[artifact contract](reference/artifacts.md), [hardware details](08_HARDWARE_AND_PERFORMANCE.md), and
[troubleshooting guide](troubleshooting.md) when reproducing a run.
