#!/usr/bin/env bash
# Execute the complete M3 load/evaluation smoke on the proposed reference GPU.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

if [[ ! -f "pyproject.toml" || ! -d "src/nanopt" ]]; then
  echo "Could not locate the NanoPT repository root." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "M3 reference smoke requires a clean checkout." >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
evidence_root="${1:-artifacts/tmp/m3-reference-${timestamp}}"
if [[ -e "${evidence_root}" ]]; then
  echo "Evidence path already exists: ${evidence_root}" >&2
  exit 2
fi
mkdir -p "${evidence_root}/data" "${evidence_root}/runs"
exec > >(tee "${evidence_root}/commands.log") 2>&1

echo "M3 reference smoke evidence: ${evidence_root}"
echo "Git commit: $(git rev-parse HEAD)"

uv sync --frozen --extra dev --extra docs

set +e
uv run nanopt doctor \
  --profile rtx_4070_ti_super_16gb \
  --json "${evidence_root}/doctor.json"
doctor_status=$?
set -e
if [[ ${doctor_status} -ne 0 && ${doctor_status} -ne 2 ]]; then
  echo "Reference doctor check failed with exit ${doctor_status}." >&2
  exit "${doctor_status}"
fi

uv run nanopt data generate \
  --output "${evidence_root}/data/tasks.jsonl" \
  --manifest "${evidence_root}/data/dataset_manifest.json"

uv run nanopt calibrate \
  --mode load \
  --device cuda

uv run nanopt calibrate \
  --mode eval \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --artifacts-root "${evidence_root}/runs" \
  --run-id calibration \
  --local-files-only \
  --device cuda

uv run nanopt eval run \
  --tasks "${evidence_root}/data/tasks.jsonl" \
  --mode deterministic \
  --checkpoint-id base \
  --artifacts-root "${evidence_root}/runs" \
  --run-id reference-base \
  --local-files-only \
  --device cuda

uv run python scripts/validate_m3_reference_smoke.py \
  "${evidence_root}" \
  --output "${evidence_root}/m3_smoke_evidence.json"

echo "M3 reference smoke passed. Review ${evidence_root}/m3_smoke_evidence.json."
