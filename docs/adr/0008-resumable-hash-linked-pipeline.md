# ADR 0008: Resumable hash-linked pipeline stages

## Status

Accepted for the M7 reference candidate.

## Decision

The official recipe runs calibration, data construction, training, evaluation, and reporting as
explicit logical stages. Ordinary model stages retain their normal child run manifests. The parent
manifest atomically records every attempt, child-manifest hash, input/output checkpoint hash, wall
time, phase memory peak, and failure/retry disclosure.

A resumed pipeline skips a completed stage only after re-hashing its retained child manifest and
output. Failed attempts are never overwritten; retries receive a stable numeric suffix.

## Why

A single opaque process is convenient while it succeeds but difficult to teach, audit, or recover.
Hash-linked stage boundaries let a learner answer which exact model produced a result and prevent a
friendly stage name from masking changed bytes.

## Consequences

- The runner is longer because orchestration is explicit.
- Every expensive stage can be inspected or resumed independently.
- Retry history and measured costs remain part of release evidence.
- The final report can be rebuilt from saved artifacts without loading a model.
