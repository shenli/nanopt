# Contributing a task or verifier

## Contract

Task content must be original or compatibly licensed, deterministic, and small enough to audit.
Training, public feedback, and protected evaluation boundaries must be explicit.

## Arithmetic/data tasks

1. Generate from a safe typed representation; never use unrestricted `eval`.
2. Record generator version, config, seed, family, split, and canonical fingerprint.
3. Keep protected splits out of training, preferences, rewards, and recipe tuning.
4. Add parser attacks and verifier-contract errors.
5. Document what capability the synthetic family does and does not measure.

## MiniSWE tasks

1. Add `task.yaml`, immutable `snapshot/`, separate `hidden_tests/`, and `oracle.patch`.
2. Restrict editable/protected globs and use trusted argv test commands.
3. Reject symlinks and compute the complete snapshot SHA-256.
4. Make the unprivileged oracle solve public and hidden tests through ordinary tools.
5. Replay the oracle from a fresh reset.
6. Test traversal, protected edits, output, timeout, and verifier isolation.

Hidden tests may be public for human audit in this educational repository, but evaluated policies
must never receive their files, paths, implementation details, or process output.
