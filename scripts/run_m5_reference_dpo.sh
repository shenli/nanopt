#!/usr/bin/env bash
# Execute M5 preference construction, DPO calibration/training, and protected comparison.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 SFT_ADAPTER_DIR [EVIDENCE_ROOT]" >&2
  exit 2
fi
sft_adapter="$1"
if [[ ! -f "${sft_adapter}/adapter_config.json" ]]; then
  echo "SFT adapter directory is invalid: ${sft_adapter}" >&2
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
  echo "M5 reference DPO requires a clean checkout." >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${2:-artifacts/tmp/m5-reference-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/data" "${evidence_root}/runs"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "M5 reference DPO evidence: ${evidence_root}"
echo "Git commit: $(git rev-parse HEAD)"
echo "SFT adapter: ${sft_adapter}"

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
"${uv_command}" run nanopt data preferences \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --output "${evidence_root}/data/preferences.jsonl" \
  --audit "${evidence_root}/data/preference_audit.json"

"${uv_command}" run nanopt calibrate \
  --mode dpo \
  --preferences "${evidence_root}/data/preferences.jsonl" \
  --sft-adapter "${sft_adapter}" \
  --limit 2 \
  --artifacts-root "${evidence_root}/runs" \
  --run-id calibration-dpo \
  --local-files-only \
  --device cuda

"${uv_command}" run nanopt train dpo \
  --preferences "${evidence_root}/data/preferences.jsonl" \
  --sft-adapter "${sft_adapter}" \
  --artifacts-root "${evidence_root}/runs" \
  --run-id dpo \
  --local-files-only \
  --device cuda

"${uv_command}" run nanopt eval run \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --mode deterministic \
  --checkpoint-id sft \
  --adapter "${sft_adapter}" \
  --adapter-name sft \
  --artifacts-root "${evidence_root}/runs" \
  --run-id sft-eval \
  --local-files-only \
  --device cuda

"${uv_command}" run nanopt eval run \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --mode deterministic \
  --checkpoint-id dpo \
  --adapter "${evidence_root}/runs/dpo/adapter/dpo" \
  --adapter-name dpo \
  --artifacts-root "${evidence_root}/runs" \
  --run-id dpo-eval \
  --local-files-only \
  --device cuda

"${uv_command}" run python -m scripts.validate_m5_reference_dpo \
  "${evidence_root}" \
  --output "${evidence_root}/m5_dpo_evidence.json"

echo "M5 reference DPO passed. Review ${evidence_root}/m5_dpo_evidence.json."
