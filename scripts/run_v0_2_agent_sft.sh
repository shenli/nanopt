#!/usr/bin/env bash
# Build, train, and evaluate the exact-token Agent SFT vertical slice on the reference host.

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
  echo "The v0.2 reference gate requires a clean checkout." >&2
  exit 2
fi

image="python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93"
if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Pinned sandbox image is missing. Pull it explicitly: docker pull ${image}" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${1:-artifacts/tmp/v0.2-agent-sft-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/runs"
evidence_root="$(cd "${evidence_root}" && pwd)"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "v0.2 Agent SFT evidence: ${evidence_root}"
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

"${nanopt_command}" agent build-sft-data \
  --tasks-root tasks/mini_swe_v1 \
  --output "${evidence_root}/data" \
  --local-files-only

"${nanopt_command}" train agent-sft \
  --dataset "${evidence_root}/data" \
  --artifacts-root "${evidence_root}/runs" \
  --run-id agent-sft \
  --local-files-only \
  --device cuda

adapter="${evidence_root}/runs/agent-sft/adapter/agent_sft"

"${nanopt_command}" agent run \
  --tasks-root tasks/mini_swe_v1 --policy model --backend docker --task-split all \
  --task-id clamp_reversed_bounds --context-policy full_transcript \
  --experiment agent_sft_eval --artifacts-root "${evidence_root}/runs" \
  --run-id base-train-task --turn-limit 2 --local-files-only --device cuda

"${nanopt_command}" agent run \
  --tasks-root tasks/mini_swe_v1 --policy model --backend docker --task-split all \
  --task-id clamp_reversed_bounds --context-policy full_transcript \
  --experiment agent_sft_eval --adapter "${adapter}" --adapter-name agent_sft \
  --artifacts-root "${evidence_root}/runs" --run-id adapted-train-task \
  --local-files-only --device cuda

"${nanopt_command}" agent run \
  --tasks-root tasks/mini_swe_v1 --policy model --backend docker --task-split all \
  --task-id clamp_reversed_bounds --context-policy observation_snapshot \
  --experiment agent_sft_eval --adapter "${adapter}" --adapter-name agent_sft \
  --artifacts-root "${evidence_root}/runs" --run-id adapted-train-task-snapshot \
  --local-files-only --device cuda

"${nanopt_command}" agent run \
  --tasks-root tasks/mini_swe_v1 --policy model --backend docker --task-split all \
  --task-id merge_without_mutation --context-policy full_transcript \
  --experiment agent_sft_eval --adapter "${adapter}" --adapter-name agent_sft \
  --artifacts-root "${evidence_root}/runs" --run-id adapted-held-out-task \
  --local-files-only --device cuda

"${python_command}" -m scripts.write_reference_checksums \
  "${evidence_root}" --output "${evidence_root}/checksums.json"
"${python_command}" -m scripts.validate_v0_2_agent_sft \
  "${evidence_root}" --output "${evidence_root}/v0.2-agent-sft-evidence.json"

echo "v0.2 Agent SFT gate passed. Review ${evidence_root}/v0.2-agent-sft-evidence.json."
