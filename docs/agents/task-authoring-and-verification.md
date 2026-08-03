# Authoring a verifiable MiniSWE task

## Learning objectives

After this chapter, you should be able to:

- describe the trusted and model-visible parts of a task;
- write a small task card with explicit edit and budget boundaries;
- explain public versus hidden verification;
- validate an oracle through the ordinary action protocol;
- recognize task designs that produce ambiguous evidence.

## Task anatomy

Every original task in `tasks/mini_swe_v1/` has four parts:

```text
task-name/
├── task.yaml       # trusted version, hashes, commands, globs, and budgets
├── snapshot/       # the only initial state copied into the episode
├── hidden_tests/   # injected only into a fresh verifier workspace
└── oracle.patch    # reviewed solution used to validate the environment
```

The suite is intentionally tiny. Each issue isolates one ordinary Python behavior so a learner can
understand the bug, visible tests, hidden edge case, and solution without first understanding a
large application.

The task card is strict and versioned. It names editable paths, protected paths, trusted test
commands, test counts, and four independent budgets. The `snapshot_sha256` covers the complete
visible starting tree. Loading or resetting a task fails if that tree changes without a deliberate
task-card update.

## Public tests teach; hidden tests judge

Public tests live inside the snapshot. The model may read and run them, but cannot modify them.
They provide fast feedback about the intended behavior. Hidden tests are copied into a different
temporary workspace only after the visible submission has been copied there. Their source, path,
and process output are omitted from observations and trajectories.

The hidden tests are present in this public educational repository so human readers can audit task
quality. “Hidden” describes the evaluation boundary: an evaluated policy must not receive those
files in its workspace or context. A serious benchmark deployment would distribute the verifier
separately as well.

## Authoring checklist

1. Write one precise issue with behavior a human can verify.
2. Keep the snapshot small and license all original content as Apache-2.0.
3. Put implementation files in `editable_globs` and tests/metadata in `protected_globs`.
4. Use argv arrays for trusted test commands; never accept a command from the policy.
5. Add public examples and genuinely distinct hidden edge cases.
6. Set bounded turns, tool calls, test runs, and wall time.
7. Compute the immutable snapshot hash and update `task.yaml`.
8. Write an oracle patch that changes only editable files.
9. Run the oracle through `apply_patch`, `run_tests`, and `finish`.
10. Replay the retained responses from a new reset and compare semantic trajectories.

A task is not accepted merely because the oracle passes. Security tests must also prove that path
traversal, symbolic links, protected-test edits, output floods, timeouts, and network attempts are
blocked or contained.

## Try the suite locally

Use the host-backed fake sandbox only for trusted fixtures and learning:

```bash
uv run nanopt agent run \
  --policy oracle \
  --backend fake \
  --tasks-root tasks/mini_swe_v1 \
  --run-id mini-swe-local
```

This command is portable but **not a security boundary**. The Linux reference run uses Docker:

```bash
uv run nanopt agent run \
  --policy oracle \
  --backend docker \
  --tasks-root tasks/mini_swe_v1 \
  --local-files-only \
  --run-id mini-swe-docker
```

Docker must already contain the exact digest-pinned image from
`configs/experiments/mini_swe_rollout.yaml`. NanoPT does not pull it implicitly because a reference
run should not silently change its execution environment.

## Interpretation

An oracle score of 1.0 proves the task is solvable through the public protocol and that its hidden
tests accept the reviewed behavior. It does not prove a language model will discover that behavior,
nor that the verifier covers every possible implementation. Retaining an unsuccessful capped model
rollout is useful evidence about the baseline policy and protocol; it is not a failed environment
validation when the oracle, isolation, and replay gates pass.
