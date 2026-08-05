# M1 completion report

Date: 2026-08-02

## Scope

M1 establishes the repository foundation only. It contains no model loading, tokenization, core
losses, data generation, evaluation, or training implementation.

## Delivered

- Python 3.11/3.12 `src/` package, `nanopt` entry point, version metadata, `pyproject.toml`, and
  `uv.lock`.
- Strict typed hardware, model, experiment, and recipe profiles with deterministic resolution,
  scalar dotted overrides, stable YAML, and leaf provenance.
- Read-only `nanopt doctor` with human and schema-versioned JSON reports.
- Atomic JSON/YAML writers, append-only JSONL streams, sanitized environment/Git metadata, run IDs,
  manifest lifecycle, and the initial run-directory contract.
- Isolated JSON structured-logging baseline.
- MkDocs Material site, MathJax rendering regression, formula lint, and strict build.
- Ruff, mypy, pytest/coverage, pre-commit, schema checks, and documented local CPU/docs validation.
- English README, contribution guide, security policy, code of conduct, citation metadata, ADRs, and
  publication checklist.

## Decisions

- ADR-0000 records the repository/distribution/import/CLI identity and Apache-2.0 license.
- ADR-0001 keeps hardware, model, and experiment profiles typed and namespaced.
- ADR-0002 uses atomic document replacement and append-only event streams.
- ADR-0003 makes local validation the project gate and removes GitHub Actions.
- Checkout profiles are bundled into the wheel so installed config resolution is independent of the
  caller's working directory.

## Verification

Commands completed in a fresh Python 3.11 environment created from `uv.lock`:

```text
ruff format --check .
ruff check .
mypy src/nanopt
pytest --cov=nanopt --cov-report=term-missing -m "not gpu and not network and not reference"
python scripts/validate_schemas.py
python scripts/lint_formulas.py docs
mkdocs build --strict
uv build
```

Results:

- 37 CPU tests passed;
- total statement/branch coverage report: 87%;
- 15 package source files passed strict mypy;
- seven JSON schemas validated and ten public YAML profiles parsed and type-checked;
- the complete documentation tree passed formula lint;
- strict documentation build passed;
- source distribution and universal wheel built;
- the wheel contains all canonical profiles and resolved `base_eval` from outside the checkout;
- real `nanopt config resolve` output and provenance were inspected;
- real `nanopt doctor` JSON validated against its schema.

The in-app visual browser was unavailable. The documentation integration test therefore builds the
site and asserts that the formula page contains Arithmatex output, the MathJax runtime script, and
the configured inline/display delimiters. This is disclosed as a visual-QA limitation, not a GPU or
browser validation claim.

## CPU and GPU status

CPU foundation validation passed on macOS arm64 with Python 3.11.15. The real doctor command returned
exit code 3 on this host because it has no CUDA device and does not match the Linux x86-64 reference
profile; that diagnostic behavior is expected and tested. No GPU smoke or reference-hardware
validation was performed or claimed for M1.

## Artifacts

Local ignored build outputs include `dist/` and `site/`. The test suite also validates a fixture run
directory against `run_manifest.schema.json`. No content was saved under `artifacts/reference/`,
because that directory is reserved for complete hardware-validation evidence.

## Deviations and next risks

The configured Python dependencies resolved to current compatible releases inside the declared
ranges and successfully imported together. M2 must pin and test exact model/tokenizer revisions,
renderer boundaries, PEFT adapter behavior, and upstream API compatibility before any model-facing
claim. Hardware settings remain `proposed_unvalidated`.
