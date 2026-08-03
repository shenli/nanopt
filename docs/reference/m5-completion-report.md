# Milestone 5 completion report

Milestone 5 is complete on the CPU and clean reference-GPU tiers. The accepted run used commit
`e8ef422ea3ca10ab7a9b3fb90daf8cf61c0edc68` and the frozen M4 SFT adapter with SHA-256
`67ecddc8970ec838f567bcb9f416b1d54475d4524d6739b9d8dce9485f2c953e`.

## Delivered vertical slice

- deterministic controlled negatives for wrong value, malformed tag, and trailing content;
- verifier-enforced chosen/rejected contracts and protected-split exclusion;
- preference fingerprints and distribution/length audit;
- FP32 frozen-reference sequence-log-probability cache with complete invalidation identity;
- exact SFT-to-DPO LoRA cloning and readable pair-weighted optimization;
- cache/live parity, held-out preference evaluation, rejection-type breakdown, and protected
  generation comparison;
- typed schemas, unit/integration tests, CPU lab, chapters, ADR, calibration, reference runner, and
  offline evidence validator.

## Reference result

The full cache contains all 80 train/validation pairs and has zero measured live/cache error. The
rejected/chosen active-token ratio was 1.088, inside the frozen $[0.75, 1.35]$ tolerance. Four DPO
optimizer steps changed held-out preference behavior as follows:

| Metric | Initial SFT copy | Final DPO |
| --- | ---: | ---: |
| Validation DPO loss | 0.693147 | 0.626070 |
| Validation chosen margin | 26.3829 | 27.7917 |
| Implicit reward accuracy | 0% by exact symmetry | 100% |

Peak reserved memory was 4,020,240,384 bytes (3.74 GiB), below the profile hard budget.

On the same 44 protected prompts, parse rate remained 95.5%. Exact accuracy changed from 86.4% for
the SFT parent to 84.1% for DPO: a 2.3-point regression, disclosed and inside the predeclared
10-point M5 review threshold. M5 demonstrates preference learning, not task-accuracy improvement.

## Rejected pilot and numerical lesson

The first complete run at commit `9670982` finished training but was rejected by the evidence
validator. The generic run schema did not yet admit DPO parent lineage. More importantly, reference
scores used separate chosen/rejected forwards while policy scores used one concatenated BF16
forward. Equal adapter weights then produced a small numerical margin difference because the batch
layouts selected different BF16 kernel paths.

The accepted implementation binds the forward layout into cache identity and uses the same layout
for reference and policy scoring. Initial held-out loss is consequently exactly $log 2$ within the
declared tolerance. This is retained as an educational example of why mathematical equivalence does
not imply bitwise numerical equivalence.

## Evidence

The reviewed compact evidence is
[`m5-reference-dpo-e8ef422.json`](evidence/m5-reference-dpo-e8ef422.json). The full ignored run
bundle remains on the reference machine at
`artifacts/tmp/m5-reference-20260803-161447` and can be regenerated with
`scripts/run_m5_reference_dpo.sh`.

This result does not validate the hardware profile. GRPO and the complete no-source-edit pipeline
remain required by M6 and M7.
