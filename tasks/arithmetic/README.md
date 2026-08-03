# NanoPT synthetic arithmetic tasks

These tasks are generated from structured arithmetic ASTs by project code. Trusted answers are
evaluated with exact rational arithmetic; the generator never calls Python `eval` and does not use
model-produced answers.

`generator_config.yaml` is the small deterministic smoke configuration. A dataset fingerprint
includes this complete configuration, parser/verifier versions, and every generated record.
Canonical AST hashes—not rendered prompt text—control split assignment and leakage checks.

The dataset is intended to teach and measure post-training mechanics in a narrow verifiable domain.
It is not evidence of broad mathematical reasoning or general assistant quality. Generated records
are covered by the repository's Apache-2.0 license.
