# Environment diagnosis

`nanopt doctor` is read-only. It reports the operating system, architecture, Python and PyTorch
versions, required and optional dependencies, CUDA runtime, visible GPUs, total and free VRAM,
compute capability, BF16/TF32 capability, Hugging Face cache location, Docker state, and hardware
profile match.

```bash
nanopt doctor --json artifacts/doctor.json
```

Exit codes are part of the CLI contract:

| Code | Meaning |
|---:|---|
| 0 | The requested validated profile matches and required capabilities are usable. |
| 2 | CUDA is usable, but the profile is unvalidated, unspecified, or does not match in non-strict mode. |
| 3 | A required dependency or usable CUDA device is missing. |
| 4 | The requested profile does not match under `--strict-profile`. |

The machine-readable output validates against `specs/schemas/doctor_report.schema.json`.
