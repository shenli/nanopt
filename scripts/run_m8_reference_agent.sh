#!/usr/bin/env bash
# Validate the M8 oracle, baseline model trajectory, and hardened Docker sandbox.

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
  echo "M8 reference evaluation requires a clean checkout." >&2
  exit 2
fi

image="python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93"
if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Pinned sandbox image is missing. Pull it explicitly: docker pull ${image}" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${1:-artifacts/tmp/m8-reference-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/runs"
evidence_root="$(cd "${evidence_root}" && pwd)"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "M8 reference agent evidence: ${evidence_root}"
echo "Git commit: $(git rev-parse HEAD)"

fresh_venv="${evidence_root}/fresh-venv"
"${uv_command}" venv "${fresh_venv}" --python 3.11
VIRTUAL_ENV="${fresh_venv}" "${uv_command}" sync \
  --active --frozen --extra dev --extra docs
nanopt_command="${fresh_venv}/bin/nanopt"
python_command="${fresh_venv}/bin/python"

"${nanopt_command}" doctor \
  --profile rtx_4070_ti_super_16gb \
  --strict-profile \
  --json "${evidence_root}/doctor.json"

"${nanopt_command}" agent run \
  --tasks-root tasks/mini_swe_v1 \
  --policy oracle \
  --backend docker \
  --task-split smoke \
  --artifacts-root "${evidence_root}/runs" \
  --run-id oracle-docker \
  --local-files-only

"${nanopt_command}" agent run \
  --tasks-root tasks/mini_swe_v1 \
  --policy model \
  --backend docker \
  --task-split smoke \
  --artifacts-root "${evidence_root}/runs" \
  --run-id model-baseline \
  --max-tasks 1 \
  --turn-limit 2 \
  --local-files-only \
  --device cuda

"${python_command}" -m scripts.run_m8_security_probes \
  --image "${image}" \
  --output "${evidence_root}/security_probes.json"

"${python_command}" -m scripts.write_reference_checksums \
  "${evidence_root}" \
  --output "${evidence_root}/checksums.json"

"${python_command}" -m scripts.validate_m8_reference_agent \
  "${evidence_root}" \
  --output "${evidence_root}/m8_agent_evidence.json"

echo "M8 reference agent evaluation passed. Review ${evidence_root}/m8_agent_evidence.json."
