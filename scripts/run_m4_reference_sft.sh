#!/usr/bin/env bash
# Execute M4 calibration, full LoRA SFT, and protected adapter evaluation.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

if [[ ! -f "pyproject.toml" || ! -d "src/nanopt" ]]; then
  echo "Could not locate the NanoPT repository root." >&2
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
  echo "M4 reference SFT requires a clean checkout." >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${1:-artifacts/tmp/m4-reference-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/data" "${evidence_root}/runs"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "M4 reference SFT evidence: ${evidence_root}"
echo "Git commit: $(git rev-parse HEAD)"

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
  --mode sft \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --limit 2 \
  --artifacts-root "${evidence_root}/runs" \
  --run-id calibration-sft \
  --local-files-only \
  --device cuda

"${uv_command}" run nanopt train sft \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --artifacts-root "${evidence_root}/runs" \
  --run-id sft \
  --local-files-only \
  --device cuda

"${uv_command}" run nanopt eval run \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --mode deterministic \
  --checkpoint-id sft \
  --adapter "${evidence_root}/runs/sft/adapter/sft" \
  --adapter-name sft \
  --artifacts-root "${evidence_root}/runs" \
  --run-id sft-eval \
  --local-files-only \
  --device cuda

"${uv_command}" run python -m scripts.validate_m4_reference_sft \
  "${evidence_root}" \
  --output "${evidence_root}/m4_sft_evidence.json"

echo "M4 reference SFT passed. Review ${evidence_root}/m4_sft_evidence.json."
