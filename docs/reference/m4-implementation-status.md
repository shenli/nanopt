# M4 implementation status

Milestone 4 is implemented on the CPU/fixture tier. Reference calibration, full LoRA training, and
protected adapter evaluation remain required before the milestone is complete.

## Delivered locally

- exact completion-only collation with no silent truncation;
- token-mean NLL and completion-token accuracy;
- explicit AdamW, active-token-correct accumulation, clipping, and cosine schedule;
- adapter/optimizer/RNG checkpoints at clean optimizer boundaries;
- deterministic resume semantics;
- `nanopt train sft`, `nanopt calibrate --mode sft`, and adapter-aware evaluation;
- strict metrics, summaries, manifests, reports, tests, chapter, and CPU lab;
- a clean-checkout reference script and offline M4 evidence validator.

The protected reference target is declared before the run: parse rate at least 50% and exact-answer
accuracy at least 5% over the same 44 held-out examples used by the M3 baseline. Lower completion
NLL is required but cannot substitute for those generation metrics.

The first four-step pilot reduced validation NLL from about 2.04 to 0.60 and generated correct
answer prefixes, but strict parse rate remained 0% because generation stopped on the tokenizer's
generic EOS rather than Qwen's chat-turn terminator. The contract now trains through `<|im_end|>`,
excludes template tokens after it, and stops generation on that same token. This correction must be
locally gated before a fresh reference run counts as evidence.

Run the reference gate with:

```bash
bash scripts/run_m4_reference_sft.sh
```

The `rtx_4070_ti_super_16gb` profile remains proposed and unvalidated regardless of this milestone
smoke; the complete pipeline validation occurs later.
