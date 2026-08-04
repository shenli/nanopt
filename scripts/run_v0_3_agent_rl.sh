#!/usr/bin/env bash
# Train and validate exact-token Mini Agent RL on the reference GPU/Docker host.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 AGENT_SFT_ADAPTER [EVIDENCE_ROOT]" >&2
  exit 2
fi
agent_sft_adapter="$1"
if [[ ! -f "${agent_sft_adapter}/adapter_config.json" ]]; then
  echo "Agent SFT adapter is missing adapter_config.json: ${agent_sft_adapter}" >&2
  exit 2
fi
agent_sft_adapter="$(cd "${agent_sft_adapter}" && pwd)"

if command -v uv >/dev/null 2>&1; then
  uv_command="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  uv_command="${HOME}/.local/bin/uv"
else
  echo "uv is required; install it from https://docs.astral.sh/uv/." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "The v0.3 reference gate requires a clean checkout." >&2
  exit 2
fi

image="python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93"
if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Pinned sandbox image is missing. Pull it explicitly: docker pull ${image}" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${2:-artifacts/tmp/v0.3-agent-rl-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/runs"
evidence_root="$(cd "${evidence_root}" && pwd)"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "v0.3 Mini Agent RL evidence: ${evidence_root}"
echo "Parent Agent SFT adapter: ${agent_sft_adapter}"
echo "Git commit: $(git rev-parse HEAD)"

fresh_venv="${evidence_root}/fresh-venv"
"${uv_command}" venv "${fresh_venv}" --python 3.11
VIRTUAL_ENV="${fresh_venv}" "${uv_command}" sync --active --frozen --extra dev --extra docs
nanopt_command="${fresh_venv}/bin/nanopt"
python_command="${fresh_venv}/bin/python"

"${nanopt_command}" doctor \
  --profile rtx_4070_ti_super_16gb \
  --strict-profile \
  --json "${evidence_root}/doctor.json"

"${nanopt_command}" train agent-rl \
  --agent-sft-adapter "${agent_sft_adapter}" \
  --tasks-root tasks/mini_swe_v1 \
  --artifacts-root "${evidence_root}/runs" \
  --run-id agent-rl \
  --local-files-only \
  --device cuda

"${python_command}" -m scripts.write_reference_checksums \
  "${evidence_root}" --output "${evidence_root}/checksums.json"
"${python_command}" -m scripts.validate_v0_3_agent_rl \
  "${evidence_root}" --output "${evidence_root}/v0.3-agent-rl-evidence.json"

echo "v0.3 Mini Agent RL gate passed. Review ${evidence_root}/v0.3-agent-rl-evidence.json."
