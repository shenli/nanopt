# ADR-0007: Synchronous exact-token GRPO with protocol termination

- Status: accepted
- Date: 2026-08-03
- Milestone: M6

## Context

The M6 reference path must expose on-policy data freshness, exact token actions, verifiable rewards,
and clipped optimization on one 16 GB GPU. Open choices include rollout termination, update epochs,
loss normalization, and whether to require a frozen-reference KL term initially.

## Decision

NanoPT samples synchronously at temperature 1 and top-p 1 with one update epoch. It stores exact
sampled token IDs and behavior log probabilities before decoding. Arithmetic episodes terminate at
generic chat EOS or the exact closing-answer protocol token sequence; the terminating IDs remain
active actions and the finish reasons remain distinct.

The reference loss uses `group_zscore` population advantages, token-mean PPO-style clipping with
$\epsilon=0.2$, and `kl_beta=0`. The DPO parent stays loaded as a frozen adapter and exact stored-ID
reference scoring is implemented for experiments with positive KL beta. The initial small recipe
uses no explicit KL term and must say so.

## Consequences

- Every update can be traced to immutable sampled actions without decode/re-tokenize ambiguity.
- Freshness and old-policy identity are simple because rollout and update alternate synchronously.
- Protocol stopping gives the verifier one complete answer while preserving its closing token.
- Token-mean normalization gives longer responses more weight and is not equivalent to
  sequence-mean; reports name the variant.
- One update epoch avoids deliberate replay but sacrifices sample reuse.
- Zero KL beta reduces single-GPU scoring cost but removes an explicit reference-drift penalty.
- Full-prefix generation is slow; it is retained as the auditable reference against which future
  cached/distributed samplers must prove parity.
