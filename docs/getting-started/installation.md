# Installation

NanoPT supports Python 3.11 and 3.12. The reference dependency workflow uses `uv.lock`:

```bash
uv sync --extra dev --extra docs
uv run nanopt --help
uv run pytest
```

These commands install the CPU development environment on any platform supported by the locked
dependencies. They do not imply reference-GPU support. CUDA and the exact reference machine are
checked separately by `nanopt doctor` and later calibration runs.

Tests marked `gpu`, `network`, or `reference` are never prerequisites for the normal CPU test job.

## Validate Milestone 1

Run the complete local gate from the repository root:

```bash
./scripts/run_m1_gate.sh
```

The command syncs the locked development environment, then checks formatting, linting, types,
tests and coverage, schemas, documentation formulas, the strict documentation build, and package
construction. It exits immediately if a check fails and prints `M1 local gate passed.` only after
all checks succeed.
