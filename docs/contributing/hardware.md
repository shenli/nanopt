# Contributing a hardware profile

## Status progression

A new profile begins `proposed_unvalidated`. A matching device name or VRAM capacity is not support.
Advance status only with the profile's complete measured protocol and a reviewed compact evidence
bundle.

## Required evidence

1. Pin OS/architecture, Python, PyTorch, CUDA/runtime, model/tokenizer revision, and lockfile.
2. Run strict `nanopt doctor`, including BF16 and available/reserved memory checks.
3. Run load, evaluation, SFT, DPO, and GRPO calibrations.
4. Execute the entire Base → SFT → DPO → GRPO recipe from a clean checkout.
5. Record phase-specific wall time and peak allocated/reserved memory.
6. Repeat final evaluation and compare exact semantic generation evidence.
7. Hash retained configs, manifests, metrics, reports, and checkpoint identities.
8. Publish only public-safe compact evidence; keep weights and personal paths out.

If any stage changes microbatch, sequence length, rollout group, dtype, or objective relative to the
profile, record the deviation and do not silently inherit the old support claim. Same-capacity GPUs
may differ in architecture, BF16 behavior, allocator pressure, kernels, thermals, and software.
