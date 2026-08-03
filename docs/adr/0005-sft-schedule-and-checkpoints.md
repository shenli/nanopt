# ADR-0005: Use cosine SFT scheduling and clean-boundary checkpoints

- Status: Accepted
- Date: 2026-08-02
- Owners: NanoPT maintainers

## Context

The SFT specification leaves the learning-rate schedule open and requires deterministic resume.
Trainer frameworks often hide both choices behind callbacks and serialized internal state, which
would undermine the repository's educational purpose.

## Decision

Use explicit linear warmup followed by cosine decay. The zero-based optimizer step determines the
learning rate, so resume does not depend on an opaque scheduler object. Build a deterministic
epoch/batch schedule from the experiment seed and allow checkpoints only after a complete optimizer
step. Save adapter weights, AdamW state, CPU/CUDA RNG state, step counts, and hashes together.

Accumulated micro-batches are weighted by active completion-token count. This preserves the stated
global token-mean objective even when response lengths differ.

## Alternatives considered

- A linear decay was reasonable but not selected; cosine is already the proposed configuration and
  keeps the endpoint visible.
- Mid-accumulation checkpoints were rejected because they require persisting partial gradients and
  dataloader position with more state than the teaching implementation justifies.
- Transformers Trainer was rejected because its callback and checkpoint machinery would hide the
  core control flow M4 is meant to teach.

## Consequences

Resume is defined only at optimizer boundaries. Changing batch size, accumulation, epochs, seed, or
maximum steps changes the deterministic schedule and invalidates a checkpoint. The implementation
can be read top to bottom and its resume fixture can compare parameters exactly.

## Validation

Unit tests must show padding invariance, zero prompt-target contribution, repeated-batch loss
reduction, adapter/optimizer integrity, and exact clean-boundary resume equivalence. The reference
gate must also stay under the configured memory budget and improve protected generation behavior.
