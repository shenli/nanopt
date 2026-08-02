# Contributing to NanoPT

NanoPT is an executable course and white-box reference implementation. Changes should make the
data flow easier to inspect and should preserve the explicit SFT, DPO, and GRPO control flow.

## Development setup

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra docs
uv run ruff format --check .
uv run ruff check .
uv run mypy src/nanopt
uv run pytest --cov=nanopt --cov-report=term-missing
uv run python scripts/validate_schemas.py
uv run python scripts/lint_formulas.py docs
uv run mkdocs build --strict
uv build
```

CPU tests must not download models. Network, GPU, reference-hardware, and security tests use the
markers defined in `pyproject.toml` and are opt-in where appropriate.

## Pull requests

Keep pull requests milestone-sized. Describe scope and non-scope, architecture decisions, commands
executed, CPU/GPU validation status, produced artifacts, deviations, and remaining risks. New core
mathematics needs hand-computable tests. New hardware claims need the complete evidence protocol.

All public code, comments, documentation, reports, and project governance must be in English.
