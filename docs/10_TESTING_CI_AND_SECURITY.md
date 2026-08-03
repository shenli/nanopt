# Testing and Security

## 1. Test pyramid

NanoPT needs unusually strong low-level tests because a one-token shift or mask bug can produce plausible-looking training curves while optimizing the wrong objective.

### 1.1 Pure unit tests

Run on CPU without network:

- causal log-probability shifting;
- action/completion masks;
- masked reductions;
- DPO loss and signs;
- group-relative advantages;
- PPO clipping for positive and negative advantages;
- KL estimators;
- parser/verifier rules;
- task generation and split leakage;
- config resolution and provenance;
- schema serialization round trips.

All core losses require at least one hand-computable fixture.

### 1.2 Model integration tests

Use a tiny locally constructed causal Transformer or a committed tiny random configuration. Tests must not download a model.

Verify:

- renderer and boundary masks;
- forward log probabilities;
- one SFT optimizer step lowers loss on a repeated mini-batch;
- one DPO step increases the chosen margin;
- sampler token log probabilities match teacher-forced scoring;
- one GRPO update moves probabilities in the expected direction on a controlled batch;
- adapter save/load produces identical logits.

### 1.3 GPU smoke tests

Marked and opt-in:

- BF16 load and forward;
- LoRA SFT step;
- DPO step with cached references;
- grouped rollout plus GRPO update;
- memory instrumentation;
- checkpoint resume.

These tests may use reduced data and sequence lengths. They are not hardware validation evidence unless run through the validation protocol.

### 1.4 End-to-end tests

- CPU synthetic micro-pipeline with a tiny model;
- GPU smoke pipeline;
- reference model pipeline on the supported hardware;
- MiniSWE environment reset, tool sequence, and hidden verification.

## 2. Property and invariant tests

Recommended properties:

- padding does not alter sequence log probabilities;
- inserting prompt-only tokens does not enter the completion loss;
- group advantages sum to approximately zero;
- all-equal group rewards generate no gradient signal;
- DPO loss is invariant to adding the same scalar to chosen and rejected log probabilities within one policy;
- cached and live reference values agree;
- reset returns the original workspace hash;
- hidden verifier results are unaffected by model-visible public-test files except through the submitted solution.

## 3. Documentation tests

The documentation build runs with strict warnings. The local validation gate must fail on:

- malformed internal links;
- missing referenced source files;
- invalid code snippets selected for execution;
- formulas using unsupported raw delimiters;
- unclosed code fences;
- generated API references that cannot import.

Formatting rule:

- inline math: `$...$`;
- display math: `$$...$$`;
- do not use raw `\(...\)` or `\[...\]` delimiters in authored pages.

A lint script should scan Markdown and exempt only pages that intentionally show forbidden syntax inside code examples.

## 4. Static analysis

Required before a milestone commit or release:

- Ruff formatting and linting;
- type checking for the main package;
- pytest with coverage report;
- JSON/YAML schema validation;
- package build and import test;
- MkDocs strict build;
- secret scanning and dependency audit where practical.

Avoid enforcing an arbitrary high total coverage percentage. Require high coverage for `core`, config, parser/verifier, and schemas, and test every documented algorithm branch.

## 5. Local validation policy

NanoPT does not use GitHub Actions. Maintainers run the documented checks locally before every
milestone commit and record the exact commands and results in the milestone completion report.

The required local gate is:

```bash
./scripts/run_m1_gate.sh
```

That wrapper stops at the first failure and runs these commands from the repository root:

```bash
uv sync --frozen --extra dev --extra docs
uv run ruff format --check .
uv run ruff check .
uv run mypy src/nanopt
uv run pytest --cov=nanopt --cov-report=term-missing
uv run python scripts/validate_schemas.py
uv run python scripts/validate_m9_curriculum.py
uv run python scripts/lint_formulas.py docs
uv run mkdocs build --strict
uv build
```

GPU smoke and reference validation run only on an explicitly selected local machine. Never execute
unreviewed pull-request code on a persistent GPU host. Store checksummed evidence only after the
complete validation protocol passes.

The separate `scripts/run_m9_curriculum_gate.sh` creates a fresh locked environment, executes every
unique CPU and systems-simulation lab declared in `specs/curriculum.yaml`, and verifies that every
reference-tier declaration points to prior passing compact evidence. It never reruns expensive GPU
training merely to validate a documentation link.

## 6. Agent-environment security tests

Required adversarial tests:

- `../` and absolute-path traversal;
- symlink escape;
- patch targeting hidden or host paths;
- oversized patch/output;
- fork bomb or process flood attempt;
- timeout and infinite loop;
- network access attempt;
- reading `/proc`, environment secrets, or mounted credentials;
- modification of public tests and task metadata;
- ANSI/control-character log injection;
- malformed JSON tool calls;
- command injection through filenames or search strings;
- container escape regression checks appropriate to the chosen runtime.

The project must state that containers reduce risk but are not a perfect boundary against hostile kernel-level attacks. Run only under the documented local threat model.

## 7. Trusted/untrusted boundaries

Trusted:

- project code on a reviewed commit;
- task generator and task metadata;
- parser and verifier;
- hidden tests;
- configuration after validation.

Untrusted:

- model output;
- generated tool arguments;
- modified workspace files;
- public pull-request code on self-hosted infrastructure;
- downloaded external task assets unless verified.

Every boundary crossing validates type, size, path, and timeout.

## 8. Dependency and model supply chain

- commit `uv.lock`;
- record package versions in manifests;
- avoid `trust_remote_code=True` in the reference model profile;
- record resolved model and tokenizer revisions;
- prefer safetensors;
- verify expected repository/model identity;
- do not execute arbitrary dataset scripts in the golden path;
- pin Docker image digests for validated agent runs;
- review dependency updates rather than auto-merging major changes.

## 9. Reproducibility tests

Exact floating-point reproducibility across drivers is not guaranteed. Required checks focus on:

- identical generated datasets for the same seed/version;
- identical config and fingerprints;
- deterministic smoke outputs where deterministic algorithms are used;
- statistically consistent reference metrics across repeated runs;
- exact checkpoint save/load behavior on the same environment.

Record PyTorch deterministic settings and known nondeterministic kernels.

## 10. Release security checklist

Before a release:

- no secrets in artifacts or git history;
- no absolute personal paths in reports;
- Docker sandbox tests pass;
- hidden tests are not packaged into model-visible task images;
- dependency audit reviewed;
- model and dataset licenses documented;
- SECURITY.md includes a reporting process and threat model;
- no unreviewed contribution is executed on a persistent GPU host.
