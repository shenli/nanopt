#!/usr/bin/env bash
# Validate every M9 chapter, local lab, and retained reference-tier evidence link.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [EVIDENCE_ROOT]" >&2
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
if [[ -n "$(git status --porcelain)" ]]; then
  echo "M9 curriculum evidence requires a clean checkout." >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${1:-artifacts/tmp/m9-curriculum-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}"
evidence_root="$(cd "${evidence_root}" && pwd)"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "M9 curriculum evidence: ${evidence_root}"
echo "Git commit: $(git rev-parse HEAD)"

fresh_venv="${evidence_root}/fresh-venv"
"${uv_command}" venv "${fresh_venv}" --python 3.11
VIRTUAL_ENV="${fresh_venv}" "${uv_command}" sync \
  --active --frozen --extra dev --extra docs

"${fresh_venv}/bin/python" -m scripts.validate_m9_curriculum \
  --execute-labs \
  --output "${evidence_root}/m9_curriculum_evidence.json"

echo "M9 curriculum gate passed. Review ${evidence_root}/m9_curriculum_evidence.json."
