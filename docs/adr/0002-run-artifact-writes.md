# ADR-0002: Use atomic documents and append-only event streams

- Status: Accepted
- Date: 2026-08-02
- Owners: NanoPT maintainers

## Context

Interrupted learning runs must remain inspectable. Configuration, environment, and manifest files
need replacement semantics, while metrics and samples need append semantics.

## Decision

JSON and YAML documents are written to a sibling temporary file, flushed, and atomically replaced.
Each JSONL event is serialized to one line and appended with one operating-system write followed by
an `fsync`. Readers report the exact malformed line if external interruption corrupts a stream.

## Alternatives considered

- In-place document writes were rejected because interruption can destroy the only manifest.
- A database was rejected as unnecessary abstraction for a local educational artifact contract.

## Consequences

Runs remain inspectable with ordinary tools. Atomic replacement assumes the temporary file and final
file share a filesystem, which the writer guarantees by creating them in the same directory.

## Validation

Tests exercise replacement, append/read round trips, partial-line diagnosis, manifest schema
validation, and failure-state persistence.
