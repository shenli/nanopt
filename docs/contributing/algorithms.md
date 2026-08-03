# Contributing an algorithm or objective variant

## Contract

An algorithm contribution must name the exact mathematical variant and keep the reference control
flow readable. “Standard DPO,” “PPO-like,” or “GRPO” is insufficient when masks, advantage scaling,
KL estimator, clipping, or reduction differ.

## Required slice

1. Define inputs, tensor shapes, dtypes, masks, reduction, and numerical domain.
2. Add a small project function with explicit intermediate values.
3. Add hand-computable values, both sign cases, invalid shapes, non-finite inputs, and gradients.
4. Add typed config only for behavior that the implementation and report consume.
5. Add a CPU lab linking scalar intuition to tensor output.
6. Add/report calibration before any full reference run.
7. Retain rejected pilots and disclose changed assumptions.

## Review questions

- Which policy generated the data, and where is its identity stored?
- Are decoded tokens ever re-tokenized as training actions?
- Which positions participate in loss and normalization?
- What is detached, frozen, or recomputed?
- Which estimator is exact, sampled, biased, or approximate?
- Can a zero/degenerate batch create invented signal?
- Which metric could improve while protected quality regresses?

Do not replace the existing white-box loop with a library trainer. Optional parity code may follow
after exact agreement on a fixed fixture.
