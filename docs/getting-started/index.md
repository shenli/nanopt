# Getting started

The foundation workflow separates configuration from environment inspection:

1. Install the locked project environment.
2. Resolve hardware, model, and experiment profiles into one immutable configuration bundle.
3. Inspect the machine with `nanopt doctor` before any model download or GPU allocation.
4. Preserve the resolved configuration, provenance, and environment report with each future run.

Training commands arrive only after their mathematical and data dependencies pass their milestone
acceptance tests.
