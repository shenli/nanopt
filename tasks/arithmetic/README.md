# NanoPT synthetic arithmetic tasks

These tasks are generated from structured arithmetic ASTs by project code. Trusted answers are
evaluated with exact rational arithmetic; the generator never calls Python `eval` and does not use
model-produced answers.

`generator_config.yaml` is the small deterministic smoke configuration. A dataset fingerprint
includes this complete configuration, parser/verifier versions, and every generated record.
Canonical AST hashes—not rendered prompt text—control split assignment and leakage checks.

`split_config.yaml` assigns all 128 generated records exactly once and records canonical hashes for
leakage checks. In M3 these named splits are smoke partitions of the current generator. Their names
reserve the future evaluation contract; they do not yet prove distinct compositional, range-shift,
or format-attack distributions.

Generate the local artifact with:

```bash
uv run nanopt data generate
```

The dataset is intended to teach and measure post-training mechanics in a narrow verifiable domain.
It is not evidence of broad mathematical reasoning or general assistant quality. Generated records
are covered by the repository's Apache-2.0 license.
