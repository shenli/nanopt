#!/usr/bin/env bash
# Build and validate v0.1 from a clean checkout and a fresh locked environment.

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
  echo "M10 release evidence requires a clean checkout." >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${1:-artifacts/tmp/m10-release-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/dist" "${evidence_root}/site"
evidence_root="$(cd "${evidence_root}" && pwd)"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "M10 release evidence: ${evidence_root}"
echo "Git commit: $(git rev-parse HEAD)"

fresh_venv="${evidence_root}/fresh-venv"
"${uv_command}" venv "${fresh_venv}" --python 3.11
VIRTUAL_ENV="${fresh_venv}" "${uv_command}" sync \
  --active --frozen --extra dev --extra docs
python_command="${fresh_venv}/bin/python"

"${fresh_venv}/bin/ruff" format --check .
"${fresh_venv}/bin/ruff" check .
"${fresh_venv}/bin/mypy" src/nanopt
"${fresh_venv}/bin/pytest" --cov=nanopt --cov-report=term-missing
"${python_command}" scripts/validate_schemas.py
"${python_command}" scripts/validate_m9_curriculum.py
"${python_command}" scripts/lint_formulas.py docs
"${fresh_venv}/bin/mkdocs" build --strict --site-dir "${evidence_root}/site"
VIRTUAL_ENV="${fresh_venv}" "${uv_command}" build --out-dir "${evidence_root}/dist"

package_venv="${evidence_root}/package-venv"
"${uv_command}" venv "${package_venv}" --python 3.11
"${uv_command}" pip install \
  --python "${package_venv}/bin/python" \
  "${evidence_root}/dist/nanopt-0.1.0-py3-none-any.whl"
"${package_venv}/bin/nanopt" --version
"${package_venv}/bin/nanopt" --help >/dev/null
"${package_venv}/bin/nanopt" config resolve \
  --config-dir configs \
  --output "${evidence_root}/package-resolved.yaml"
set +e
"${package_venv}/bin/nanopt" doctor --json "${evidence_root}/package-doctor.json"
doctor_status=$?
set -e
# Exit 3 is the documented result when a complete installation has no usable CUDA device.
if [[ ${doctor_status} -ne 0 && ${doctor_status} -ne 2 && ${doctor_status} -ne 3 ]]; then
  echo "Installed-wheel doctor smoke failed with exit ${doctor_status}." >&2
  exit "${doctor_status}"
fi

"${python_command}" -m scripts.validate_m10_release \
  --dist-dir "${evidence_root}/dist" \
  --output "${evidence_root}/m10_release_evidence.json"

echo "M10 release gate passed. Review ${evidence_root}/m10_release_evidence.json."
