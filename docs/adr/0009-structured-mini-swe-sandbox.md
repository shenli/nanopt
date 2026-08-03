# ADR-0009: Structured MiniSWE tools with a hardened Docker reference backend

- Status: Accepted
- Date: 2026-08-03

## Context

M8 needs a stateful coding environment that learners can read and replay. Model output and modified
workspace code are untrusted, while v0.1 must remain small enough to run locally. A generic shell
tool would make the action space difficult to validate and would give prompt instructions the job
of enforcing security. A host-only fake backend is useful for tests but is not isolation.

## Decision

NanoPT accepts exactly one strict JSON action from an allow-listed union: list files, read a file,
search text, apply a bounded unified diff, run the task-selected public tests, or finish. All paths
cross one workspace policy boundary. Test commands come only from reviewed task cards.

The reference backend uses a digest-pinned official Python Docker image. It runs as a numeric
non-root user with no network, GPU, Linux capabilities, privilege escalation, writable root
filesystem, or Docker socket, and with explicit memory, swap, PID, CPU, timeout, and output limits.
Hidden verification copies the visible submission into a fresh workspace and injects hidden tests
there without returning their source or output.

The in-process fake backend remains available only for trusted unit tests and CPU teaching labs. It
must be labeled non-secure in artifacts and documentation.

## Consequences

- Model actions are small, typed, bounded, and replayable.
- Docker and the exact pre-pulled image are required for reference security claims.
- The environment supports editing existing UTF-8 files, not arbitrary package installation,
  file creation/deletion, or shell workflows.
- Hidden tests are auditable by repository readers but absent from evaluated-policy workspaces.
- Containers reduce risk but do not promise protection from kernel/daemon compromise.
- M8 evaluates policies only; Agent SFT and Agent RL remain later milestones.
