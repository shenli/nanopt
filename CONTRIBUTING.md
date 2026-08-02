# Contributing to NanoPT

NanoPT is an executable course and white-box reference implementation. Changes should make the
data flow easier to inspect and should preserve the explicit SFT, DPO, and GRPO control flow.

## Development setup

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
./scripts/run_m1_gate.sh
```

The script installs the locked development dependencies and executes every required M1 check. The
individual commands are listed in
[`docs/10_TESTING_CI_AND_SECURITY.md`](docs/10_TESTING_CI_AND_SECURITY.md#5-local-validation-policy).

CPU tests must not download models. Network, GPU, reference-hardware, and security tests use the
markers defined in `pyproject.toml` and are opt-in where appropriate.

## Educational code style

NanoPT is meant to be read while it runs. Public functions and classes should have docstrings that
state their contract. Core tensor functions must also document tensor shapes, mask meaning, and the
reduction used. Add comments where they explain a design choice, invariant, precedence rule, or
numerical subtlety. Do not add comments that merely translate the next line of Python into English.

Keep algorithm entry points readable from top to bottom. Prefer small named functions and explicit
intermediate values over callbacks, factories, or compressed expressions. Tests for core math
should use tiny hand-computable examples and explain the expected result.

Repository-specific instructions for coding agents are kept in [`AGENTS.md`](AGENTS.md). They also
define the rule that implementation, hand-computable tests, and learner-facing tutorials move
together in one change.

## Pull requests

Keep pull requests milestone-sized. Describe scope and non-scope, architecture decisions, commands
executed, CPU/GPU validation status, produced artifacts, deviations, and remaining risks. New core
mathematics needs hand-computable tests. New hardware claims need the complete evidence protocol.

All public code, comments, documentation, reports, and project governance must be in English.
