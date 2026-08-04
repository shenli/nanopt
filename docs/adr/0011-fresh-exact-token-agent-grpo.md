# ADR-0011: Fresh exact-token Agent GRPO

**Status:** Accepted for v0.3.

## Context

Stateful agent episodes contain multiple prompts because each tool changes the observation. A
single concatenated completion would obscure those boundaries. Replaying older episodes may save
rollout cost, but policy drift makes a clipped on-policy objective harder to interpret. Sparse
hidden rewards also leave several defensible credit-assignment choices.

## Decision

NanoPT v0.3 collects grouped episodes from independent resets of one immutable task snapshot. Each
action stores its exact online prompt IDs, sampled IDs, mask, and FP32 behavior log probabilities.
The terminal hidden-verifier score becomes a group-relative episode advantage and is assigned to
every sampled action token. Updates accept only the current collection policy version and use one
clipped update epoch. The oldest and newest groups are rescored after training for a non-training
staleness study. Terminal-only credit is a token-coverage ablation, not another claimed optimizer.
The reference run evaluates every post-update policy on the task-level validation split and
publishes the highest-reward boundary, breaking ties toward the earliest version. The untrained
parent is recorded as a baseline but is not eligible for selection.

## Consequences

- Decode/re-tokenize cannot enter the Agent RL training path.
- Hidden reward is revealed only after termination and never becomes an observation.
- Equal-reward groups produce honest zero advantages.
- Validation-boundary selection prevents a later sparse-reward update from silently replacing a
  stronger earlier Agent RL policy; it does not turn the validation task into a test claim.
- v0.3 favors interpretability over sample reuse and throughput.
- Partial rollout scheduling, replay buffers, policy-lag corrections, and accelerated generation
  remain v0.4 work.
