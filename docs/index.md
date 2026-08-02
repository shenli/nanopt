# NanoPT

NanoPT is an executable course and white-box reference implementation for post-training a small
base language model. The project connects hand-computable objectives to tested tensor functions,
single-GPU experiments, inspectable artifacts, and a secure stateful agent environment.

Milestone 1 is the repository foundation. Milestone 2 begins the tested mathematical core, without
training code yet. Start with the [prerequisites](getting-started/prerequisites.md), install the
project, then continue to tokens, masks, and causal log probabilities.

## Status vocabulary

- **Required** means the release contract requires the behavior.
- **Proposed** means the setting is an explicit starting point that still needs calibration.
- **Validated** means a reviewed evidence bundle passed the complete reference protocol.

The RTX 4070 Ti SUPER configuration is currently proposed and unvalidated.
