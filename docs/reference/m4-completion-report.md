# M4 completion report

Milestone 4 is complete. Completion-only LoRA SFT passed the CPU/fixture gate, reference
calibration, full proposed training run, and protected adapter evaluation. The result demonstrates
format and instruction learning relative to the frozen M3 base baseline; it does **not** validate
the later DPO/GRPO pipeline or the hardware profile as a whole.

## Delivered

- exact prompt-prefix rendering and completion-only right-padded collation;
- token-mean completion NLL and completion-token accuracy;
- explicit AdamW, active-token-correct accumulation, gradient clipping, and cosine schedule;
- adapter/optimizer/RNG checkpoints only at clean optimizer boundaries;
- deterministic resume semantics and checkpoint hash validation;
- `nanopt train sft`, `nanopt calibrate --mode sft`, and adapter-aware evaluation;
- teacher-forced metrics, offline reports, protected generation evaluation, and lineage manifests;
- an educational chapter, CPU lab, ADR, reference script, and offline evidence validator.

## Local acceptance evidence

The local gate covers prompt-target gradient exclusion, padding invariance, repeated-batch loss
reduction, active-token accumulation, deterministic schedule endpoints, adapter/optimizer tamper
detection, exact clean-boundary resume equivalence, and a tiny end-to-end Qwen/LoRA run. The strict
real-token boundary distinguishes Qwen's generic EOS, chat terminator, and the task protocol's exact
multi-token stop sequence.

## Reference SFT evidence

The clean reference gate passed on commit
`05bd73a0236625f58b14853e1489cea1e977d43a`:

```bash
bash scripts/run_m4_reference_sft.sh
```

| Field | Observed value |
| --- | --- |
| Host | Linux x86-64 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, compute capability 8.9 |
| Driver / CUDA / PyTorch | 560.35.03 / 12.6 / 2.7.1+cu126 |
| Model revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Dataset | 128 records; 64 train, 16 validation, 44 protected evaluation |
| Calibration | 2 train examples, 1 optimizer step, explicitly non-representative |
| Full SFT | 4 optimizer steps, BF16 LoRA, 64 training examples |
| Peak reserved memory | 3,070,230,528 bytes (2.86 GiB) |
| Adapter SHA-256 | `67ecddc8970ec838f567bcb9f416b1d54475d4524d6739b9d8dce9485f2c953e` |

Validation completion NLL fell from 2.0536 to 0.6211. Completion-token accuracy rose from 78.49%
to 90.78%. These are teacher-forced diagnostics, so the gate separately evaluated generated text.

## Protected generation result

The targets were declared before the final run: at least 50% parse rate and 5% exact-answer
accuracy over the same 44 held-out examples as M3.

| Metric | Base (M3) | SFT (M4) |
| --- | ---: | ---: |
| Strict parse rate | 0/44 (0%) | 42/44 (95.45%) |
| Exact-answer accuracy | 0/44 (0%) | 38/44 (86.36%) |
| Exact protocol stop | not configured | 43/44 (97.73%) |
| Length limit | 44/44 (100%) | 1/44 (2.27%) |

The SFT accuracy 95% Wilson interval is 73.29–93.60%; the parse-rate interval is 84.87–98.74%.
The compact reviewed validator output is preserved in
[the M4 reference evidence summary](evidence/m4-reference-sft-05bd73a.json).

## What the pilots taught us

The original four-step adapter already generated correct `<solution>` and `<answer>` prefixes, but
evaluation continued after the closing answer tag and the strict parser rejected trailing content.
The investigation exposed two distinct boundaries:

1. Qwen chat turns end with `<|im_end|>`, not the tokenizer's generic `<|endoftext|>` EOS.
2. Arithmetic tasks have a public exact `</answer>` token stop sequence.

NanoPT now trains through the chat terminator, excludes the template newline after termination, and
stops task evaluation immediately after preserving the complete closing-tag token sequence. The
parser was not weakened, and the checked-in four-step training profile did not change.

## Remaining boundary

The hardware profile remains `proposed_unvalidated`. M4 did not test DPO, GRPO, full-pipeline
resume, throughput targets, or long-run thermal behavior. Those claims remain gated by their later
milestones. The next stage is M5: controlled preference construction and the readable DPO loop.
