# Extending NanoPT without hiding the lesson

## Learning objectives

After this chapter, you should be able to:

- scope an extension as one readable vertical slice;
- identify which tests, docs, configs, and evidence must move together;
- avoid expanding hardware or algorithm claims without validation;
- choose an extension path appropriate for v0.2 or later.

## The vertical-slice rule

An extension is complete when a learner can connect:

```text
motivation → explicit contract → readable implementation → hand test
→ runnable lab → measured evidence → limitation
```

Do not begin with a generic trainer interface. Begin with the smallest typed record and numerical or
environment invariant that makes the new behavior distinguishable.

## Supported paths

- **Algorithm:** add a named objective/reduction variant with hand-computable sign, mask, and
  gradient tests. See [algorithm contributions](../contributing/algorithms.md).
- **Task/verifier:** add original licensed snapshots, strict task cards, an unprivileged oracle,
  public/hidden isolation, and attack tests. See [task contributions](../contributing/tasks.md).
- **Hardware:** add a proposed profile first, then complete calibration and full evidence before
  changing support status. See [hardware contributions](../contributing/hardware.md).
- **Agent SFT/RL:** preserve exact response token IDs, observation/action boundaries, reset hashes,
  policy versions, and hidden-verifier isolation. These are v0.2/v0.3 projects, not v0.1 claims.
- **Rollout backend:** prove exact sampled IDs/log probabilities and termination parity before
  comparing throughput.

## Capstone exercise

Choose one path and write a one-page proposal containing scope, non-scope, invariant, smallest test,
lab tier, expected artifacts, security/data risks, and claim that would remain unsupported. Run the
existing local gate before writing implementation code so failures have a known baseline.

## Common mistakes

- Adding an option that no report or test records.
- Copying benchmark data without checking license/leakage.
- Replacing readable control flow with a callback abstraction.
- Calling one smoke run validated hardware support.
- Publishing only successful metrics and deleting rejected pilots.

## Industrial mapping

Production frameworks need plugin layers, distributed fault tolerance, access controls, and broader
compatibility. NanoPT accepts some duplication to keep one algorithm readable top to bottom. An
extension may add infrastructure, but it must not remove the white-box reference path.
