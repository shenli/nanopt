#!/usr/bin/env bash
# Build a fresh locked environment, regenerate data, and run the complete M7 recipe.

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
  echo "M7 reference pipeline requires a clean checkout." >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${1:-artifacts/tmp/m7-reference-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/data" "${evidence_root}/pipelines"
evidence_root="$(cd "${evidence_root}" && pwd)"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "M7 reference pipeline evidence: ${evidence_root}"
echo "Git commit: $(git rev-parse HEAD)"

fresh_venv="${evidence_root}/fresh-venv"
"${uv_command}" venv "${fresh_venv}" --python 3.11
VIRTUAL_ENV="${fresh_venv}" "${uv_command}" sync \
  --active --frozen --extra dev --extra docs
nanopt_command="${fresh_venv}/bin/nanopt"
python_command="${fresh_venv}/bin/python"

set +e
"${nanopt_command}" doctor \
  --profile rtx_4070_ti_super_16gb \
  --json "${evidence_root}/doctor.json"
doctor_status=$?
set -e
if [[ ${doctor_status} -ne 0 && ${doctor_status} -ne 2 ]]; then
  echo "Reference doctor check failed with exit ${doctor_status}." >&2
  exit "${doctor_status}"
fi

"${nanopt_command}" data generate \
  --output "${evidence_root}/data/tasks.jsonl" \
  --manifest "${evidence_root}/data/dataset_manifest.json"

"${nanopt_command}" pipeline run \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --recipe math_pipeline \
  --artifacts-root "${evidence_root}/pipelines" \
  --run-id reference \
  --local-files-only \
  --device cuda

"${python_command}" -m scripts.write_reference_checksums \
  "${evidence_root}" \
  --output "${evidence_root}/checksums.json"

"${python_command}" -m scripts.validate_m7_reference_pipeline \
  "${evidence_root}" \
  --output "${evidence_root}/m7_pipeline_evidence.json"

echo "M7 reference pipeline passed. Review ${evidence_root}/m7_pipeline_evidence.json."
