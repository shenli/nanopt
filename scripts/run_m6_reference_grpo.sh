#!/usr/bin/env bash
# Execute M6 GRPO calibration, full exact-token training, and protected comparison.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DPO_ADAPTER_DIR [EVIDENCE_ROOT]" >&2
  exit 2
fi
dpo_adapter="$1"
if [[ ! -f "${dpo_adapter}/adapter_config.json" ]]; then
  echo "DPO adapter directory is invalid: ${dpo_adapter}" >&2
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
  echo "M6 reference GRPO requires a clean checkout." >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${2:-artifacts/tmp/m6-reference-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/data" "${evidence_root}/runs"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "M6 reference GRPO evidence: ${evidence_root}"
echo "Git commit: $(git rev-parse HEAD)"
echo "DPO adapter: ${dpo_adapter}"

"${uv_command}" sync --frozen --extra dev --extra docs

set +e
"${uv_command}" run nanopt doctor \
  --profile rtx_4070_ti_super_16gb \
  --json "${evidence_root}/doctor.json"
doctor_status=$?
set -e
if [[ ${doctor_status} -ne 0 && ${doctor_status} -ne 2 ]]; then
  echo "Reference doctor check failed with exit ${doctor_status}." >&2
  exit "${doctor_status}"
fi

"${uv_command}" run nanopt data generate \
  --output "${evidence_root}/data/tasks.jsonl" \
  --manifest "${evidence_root}/data/dataset_manifest.json"

"${uv_command}" run nanopt calibrate \
  --mode grpo \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --dpo-adapter "${dpo_adapter}" \
  --artifacts-root "${evidence_root}/runs" \
  --run-id calibration-grpo \
  --local-files-only \
  --device cuda

"${uv_command}" run nanopt train grpo \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --dpo-adapter "${dpo_adapter}" \
  --artifacts-root "${evidence_root}/runs" \
  --run-id grpo \
  --local-files-only \
  --device cuda

"${uv_command}" run nanopt eval run \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --mode deterministic \
  --checkpoint-id dpo \
  --adapter "${dpo_adapter}" \
  --adapter-name dpo \
  --artifacts-root "${evidence_root}/runs" \
  --run-id dpo-eval \
  --local-files-only \
  --device cuda

"${uv_command}" run nanopt eval run \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --mode deterministic \
  --checkpoint-id grpo \
  --adapter "${evidence_root}/runs/grpo/adapter/grpo" \
  --adapter-name grpo \
  --artifacts-root "${evidence_root}/runs" \
  --run-id grpo-eval \
  --local-files-only \
  --device cuda

"${uv_command}" run python -m scripts.validate_m6_reference_grpo \
  "${evidence_root}" \
  --output "${evidence_root}/m6_grpo_evidence.json"

echo "M6 reference GRPO passed. Review ${evidence_root}/m6_grpo_evidence.json."
