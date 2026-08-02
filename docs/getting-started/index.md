# Getting started

The foundation workflow separates configuration from environment inspection:

1. Review the prerequisites and run the CPU readiness check.
2. Install the locked project environment.
3. Resolve hardware, model, and experiment profiles into one immutable configuration bundle.
4. Inspect the machine with `nanopt doctor` before any model download or GPU allocation.
5. Preserve the resolved configuration, provenance, and environment report with each future run.

Training commands arrive only after their mathematical and data dependencies pass their milestone
acceptance tests.
