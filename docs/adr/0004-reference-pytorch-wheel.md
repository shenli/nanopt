# ADR-0004: Pin the Linux reference PyTorch wheel

- Status: Accepted
- Date: 2026-08-02
- Owners: NanoPT maintainers

## Context

NanoPT originally allowed any PyTorch release from 2.6 up to 3. A fresh lock on the Linux reference
host resolved PyTorch 2.13 with CUDA 13 libraries. The host's NVIDIA 560.35.03 driver could see the
RTX 4070 Ti SUPER, but PyTorch reported that the driver was too old and exposed zero CUDA devices.
The failure happened before model loading, so it was a software-stack mismatch rather than a model
memory failure.

An educational repository should make this boundary visible. A GPU model name alone is not enough:
the application, PyTorch wheel, CUDA generation, and NVIDIA driver must form a compatible stack.

## Decision

Pin PyTorch 2.7.1. For Linux x86-64 installs made with `uv`, resolve `torch` from PyTorch's CUDA 12.6
index. Let other platforms resolve the same PyTorch version from PyPI. Keep both platform branches
in `uv.lock` so a clean checkout produces the intended environment without manual package surgery.

The reference gate must still run `nanopt doctor`; the package pin is not a substitute for measuring
the actual host. Upgrading the reference host's driver or moving to a newer CUDA generation requires
re-running the hardware evidence protocol and revisiting this ADR.

## Alternatives considered

- Upgrading the host driver was deferred because it changes machine-wide state and was unnecessary
  for Milestone 3.
- Keeping the broad PyTorch range was rejected because a routine lock refresh could silently change
  the required driver generation.
- Using a CPU-only Linux wheel was rejected because the reference gate must execute the real model
  on CUDA.

## Consequences

Linux x86-64 `uv` environments download the larger CUDA-enabled wheel even when no GPU is present.
Installers that ignore `[tool.uv.sources]` do not receive the reference index selection, so the
documented reproducible path is `uv sync --frozen`. Dependency updates must be tested on macOS/CPU
and on the Linux reference host before changing the pin.

## Validation

The local gate must pass from the refreshed lock. On the reference host, `nanopt doctor` must report
the expected Linux/x86-64 platform, CUDA availability, one RTX 4070 Ti SUPER, and the recorded
PyTorch/CUDA/driver versions before any model evaluation result can count as evidence.
