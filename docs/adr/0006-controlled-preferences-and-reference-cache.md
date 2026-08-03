# ADR-0006: Controlled preference failures and a precomputed SFT reference cache

- Status: accepted
- Date: 2026-08-03
- Milestone: M5

## Context

DPO needs rejected completions and frozen-reference log probabilities. The roadmap leaves two real
choices open: whether negatives come from uncontrolled model sampling or declared transformations,
and whether a second reference adapter remains live during every update or is scored once.

The reference path must remain educational, deterministic, and viable on one 16 GB GPU.

## Decision

NanoPT v0.1 uses three deterministic rejection transformations: canonical wrong answer, malformed
answer tag, and trailing content. Every chosen completion must pass the strict verifier and every
rejected completion must produce its declared failure. Only train and validation tasks are eligible.

The frozen SFT policy is scored once into an FP32 cache. Its identity binds model/tokenizer
revisions, SFT adapter content, renderer/template, preference data, length policy, EOS policy, and
sum reduction. It also binds the chosen/rejected forward layout so the cached reference and live
policy take the same BF16 numerical path. The DPO policy is then created by an exact in-memory clone
of the SFT adapter. A live cache-parity sample must pass before the first optimizer step.

## Consequences

- Pair causality and failure modes are easy to inspect and reproduce.
- Protected prompts cannot leak through offline preference construction.
- Reference inference and trainable-policy activations need not coexist during updates.
- The initial DPO loss provides a useful exact-copy check at $log 2$.
- The small controlled mixture does not approximate the diversity or ambiguity of human preference
  data; reports must not generalize beyond the arithmetic task.
- Cache invalidation is stricter and more verbose than accepting an arbitrary score file, which is
  intentional because a stale score silently changes the objective.
