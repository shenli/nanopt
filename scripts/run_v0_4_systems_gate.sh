#!/usr/bin/env bash
# Run the deterministic v0.4 systems reference gate from a clean locked checkout.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "The v0.4 systems gate requires a clean checkout." >&2
  exit 2
fi

if command -v uv >/dev/null 2>&1; then
  uv_command="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  uv_command="${HOME}/.local/bin/uv"
else
  echo "uv is required; install it from https://docs.astral.sh/uv/." >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${1:-artifacts/tmp/v0.4-systems-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/runs"
evidence_root="$(cd "${evidence_root}" && pwd)"

"${uv_command}" sync --frozen --extra dev --extra docs
"${uv_command}" run nanopt systems simulate \
  --artifacts-root "${evidence_root}/runs" \
  --run-id resumable-rollouts
"${uv_command}" run python -m scripts.validate_v0_4_systems \
  "${evidence_root}/runs/resumable-rollouts" \
  --output "${evidence_root}/v0.4-systems-evidence.json"

echo "v0.4 systems gate passed. Review ${evidence_root}/v0.4-systems-evidence.json."
