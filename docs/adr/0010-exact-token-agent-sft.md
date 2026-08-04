# ADR-010: Freeze exact-token, replay-linked Agent SFT examples

**Status:** Accepted for v0.2

## Context

Agent demonstrations contain multiple observations and actions. Reconstructing prompts inside a
trainer can silently change chat-template boundaries, duplicate transcript content, or train on
previous actions again.

## Decision

NanoPT freezes unpadded token IDs, attention masks, current-action masks, prompt length, messages,
context policy, chat-template hash, and source-trajectory lineage before training. Source episodes
must pass public and hidden verification and replay exactly. Training consumes stored IDs without
decoding and re-tokenizing them.

Generation stops when the decoded prefix is exactly one complete JSON object. The strict action
parser remains the trust boundary; the stop condition does not repair malformed output.

## Consequences

Datasets are larger and tokenizer-specific, but their training semantics are inspectable. Context
policy changes require a new dataset. Teacher-forced gains remain separate from Docker task scores.
