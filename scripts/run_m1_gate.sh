#!/usr/bin/env bash
# Run the complete Milestone 1 validation contract from the repository root.

set -euo pipefail

if [[ ! -f "pyproject.toml" || ! -d "src/nanopt" ]]; then
  echo "Run this script from the NanoPT repository root." >&2
  exit 2
fi

run_step() {
  local label="$1"
  shift
  printf '\n==> %s\n' "$label"
  "$@"
}

# Keep this list in the same order as the local gate documented in
# docs/10_TESTING_CI_AND_SECURITY.md. The script stops at the first failure.
run_step "Sync locked development and documentation dependencies" \
  uv sync --frozen --extra dev --extra docs
run_step "Check formatting" uv run ruff format --check .
run_step "Run Ruff lint rules" uv run ruff check .
run_step "Run strict type checking" uv run mypy src/nanopt
run_step "Run CPU tests with coverage" \
  uv run pytest --cov=nanopt --cov-report=term-missing
run_step "Validate schemas and configuration files" \
  uv run python scripts/validate_schemas.py
run_step "Validate curriculum manifest" \
  uv run python scripts/validate_m9_curriculum.py
run_step "Lint documentation formulas" \
  uv run python scripts/lint_formulas.py docs
run_step "Build documentation in strict mode" uv run mkdocs build --strict
run_step "Build source and wheel distributions" uv build

printf '\nM1 local gate passed.\n'
