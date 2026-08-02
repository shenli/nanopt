# Instructions for coding agents

NanoPT is an executable course and a white-box reference implementation. Optimize for a learner's
understanding before optimizing for abstraction or brevity.

## Educational standard

- Keep algorithm entry points readable from top to bottom.
- Use small, explicitly named functions and intermediate values.
- Give public functions and classes contract-focused docstrings.
- For tensor functions, document shapes, dtypes, coordinate systems, mask meaning, reductions, and
  numerical precision.
- Comment design choices, invariants, boundary shifts, and numerical subtleties. Do not comment
  obvious syntax.
- Keep core SFT, DPO, GRPO, and RLVR mathematics in project code rather than hiding it behind a
  trainer framework.
- Preserve exact sampled token IDs for policy-gradient training. Never decode and re-tokenize a
  training trajectory.

## Code, tests, and course material move together

Every new mathematical concept or vertical slice must include:

1. readable implementation code;
2. tiny hand-computable unit tests, including failure cases;
3. learner-facing English documentation that explains intuition, equations, tensor shapes, and
   common mistakes;
4. a CPU lab or runnable example when the concept does not require a model or GPU.

Tutorials should link to the canonical source and tests instead of pasting a second implementation.
When code behavior changes, update the corresponding tutorial in the same change.

## Correctness and scope

- Follow the milestone dependency order in `docs/12_IMPLEMENTATION_ROADMAP.md`.
- Follow tensor and objective contracts in `docs/05_ALGORITHM_SPECIFICATIONS.md`.
- Treat masks as the source of truth; do not bury masking rules in `-100` labels.
- Compute log-softmax and loss reductions in FP32 unless a specification explicitly says otherwise.
- Reject ambiguous shapes and zero-active-token reductions with clear errors.
- Do not claim GPU or hardware support without the required evidence bundle.
- Do not add network access to CPU tests.

Before handing off a milestone slice, run `./scripts/run_m1_gate.sh` plus any relevant lab, GPU, or
reference checks. Record unavailable hardware checks honestly rather than weakening the claim.
