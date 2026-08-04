# Installation

NanoPT supports Python 3.11 and 3.12. The reference dependency workflow uses `uv.lock`:

```bash
uv sync --frozen --extra dev --extra docs
uv run nanopt --help
uv run pytest
```

On macOS and other non-reference platforms, PyTorch comes from PyPI. On Linux x86-64, `uv` selects
the pinned PyTorch 2.7.1 CUDA 12.6 wheel from PyTorch's package index. The platform split matters:
the reference host's NVIDIA 560 driver can run CUDA 12.6 binaries, while the newer CUDA 13 wheel
selected by an unconstrained dependency could not initialize CUDA on that machine. See
[ADR-0004](../adr/0004-reference-pytorch-wheel.md) for the decision and trade-offs.

Installing a CUDA-enabled wheel does not by itself prove GPU support. `nanopt doctor` still checks
the actual driver, CUDA runtime, device, compute capability, and memory before a reference run.

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
