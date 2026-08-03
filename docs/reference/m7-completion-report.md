# Milestone 7 completion report

Milestone 7 is complete. A clean candidate commit ran the entire calibrated Base → SFT → DPO →
GRPO recipe in a fresh environment on the RTX 4070 Ti SUPER reference host, with no source edits
between stages. The offline validator accepted the parent/child lineage, protected-data boundary,
measured memory, repeated evaluation, and evidence checksums.

## Reproduce the protocol

From a clean checkout with the pinned model already cached:

```bash
bash scripts/run_m7_reference_pipeline.sh
```

The script creates a fresh Python 3.11 environment from `uv.lock`, runs `nanopt doctor`, regenerates
the 128-task dataset, executes all 15 stages, writes 177 retained-file checksums, and runs the M7
validator. The reference run used commit
`92564f3abd3bfa353ab92587ddbc09892c9b13de`.

## Protected comparison

All checkpoints used the same 44 frozen protected tasks and deterministic evaluation path.

| Checkpoint | Exact accuracy | Parse rate |
| --- | ---: | ---: |
| Base | 0.0% | 0.0% |
| SFT | 86.4% | 95.5% |
| DPO | 84.1% | 95.5% |
| GRPO | 88.6% | 95.5% |

The DPO regression relative to SFT remains disclosed. GRPO recovered that regression and finished
2.3 percentage points above SFT and 4.5 points above DPO. The final GRPO evaluation was repeated;
token IDs, decoded responses, parser states, verifier states, and finish reasons matched exactly.
Run identity and measured generation time were deliberately excluded from the semantic comparison.

## Runtime and memory

- Retained stage wall time: 222.68 seconds.
- Peak stage: GRPO, 7,954,497,536 bytes (7.41 GiB) reserved.
- Profile hard limit: 15.2 GiB reserved.
- Failed or retried stage attempts: 0.
- Final GRPO adapter SHA-256:
  `00ce1095a4f6863aafcdc77b68f69385197b339454bb26687974ba4cf1e038a2`.

Doctor recorded NVIDIA driver `560.35.03`, CUDA runtime `12.6`, compute capability 8.9, and BF16
runtime support. Every load, evaluation, SFT, DPO, and GRPO calibration executed on CUDA.

## Reproducibility and lineage

The parent manifest contains one record for every logical stage, the retained attempt, child
manifest hash, input/output checkpoint hashes, stage wall time, and phase memory peak. Resume
re-hashes completed outputs. Controlled preference construction includes only train and validation
splits; protected test records were not used for training, reward shaping, or recipe selection.

The complete ignored working bundle remains on the reference host. The compact reviewed evidence is
[`m7-reference-pipeline-92564f3.json`](evidence/m7-reference-pipeline-92564f3.json). It binds the
pipeline manifest and 177-file checksum manifest without publishing machine-specific paths or model
weights.

## Support decision

The complete M7 and hardware-validation gates passed. The
`rtx_4070_ti_super_16gb` profile, the three reference training experiments, and the official recipe
are now marked `validated` by the same commit that publishes this evidence. This support claim is
specific to the recorded Linux x86-64, RTX 4070 Ti SUPER, pinned model, and locked software path; it
does not imply support for every 16 GB GPU.
