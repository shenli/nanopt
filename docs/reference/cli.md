# CLI reference

NanoPT exposes only commands backed by implemented, tested behavior:

```text
nanopt doctor
nanopt config resolve
nanopt data generate
nanopt data preferences
nanopt calibrate --mode load|eval|sft|dpo|grpo
nanopt train sft
nanopt train dpo
nanopt train grpo
nanopt eval run [--adapter ADAPTER_DIR]
nanopt pipeline run --tasks TASKS --recipe math_pipeline
nanopt agent run --policy oracle|model --backend docker|fake
nanopt report build RUN_DIR
nanopt artifacts inspect RUN_DIR
```

`nanopt data generate` writes the deterministic arithmetic task JSONL and split manifest. It
refuses to append to a non-empty output, preventing two dataset versions from being mixed.
`nanopt data preferences` derives only train/validation pairs, verifies every controlled failure,
and writes a fingerprinted audit beside the pair JSONL.

`nanopt eval run` creates a run directory before loading the model, appends each example before
aggregation, and builds Markdown/HTML reports. Use `--mode deterministic` for one greedy completion
per task or `--mode sampled` for the profile's fixed sample count and seed schedule. The task JSONL
must have a sibling `dataset_manifest.json`; counts and canonical hashes are verified before model
loading.

`nanopt calibrate --mode load` exercises the exact model-loader path. `--mode eval` requires a task
JSONL and uses an explicit small limit; its manifest labels the result non-representative.

`nanopt calibrate --mode sft` runs one deliberately non-representative optimizer step. `nanopt train
sft` executes the configured completion-only LoRA stage and may resume from a clean-boundary
checkpoint with `--resume-from`. The run writes teacher-forced metrics and a report, but protected
generation still requires `nanopt eval run --adapter ADAPTER_DIR --checkpoint-id sft`.

`nanopt calibrate --mode dpo` exercises reference-cache construction, live/cache parity, exact SFT
adapter cloning, forward/backward, and one short optimization path. `nanopt train dpo` runs the full
configured pair set and writes the cache manifest, per-type breakdown, final adapter, and report.
Both require `--preferences` and `--sft-adapter`.

`nanopt calibrate --mode grpo` performs grouped generation, decoding and strict reward, advantage
calculation, exact-token collation, backward, clipping, and an optimizer step. `nanopt train grpo`
alternates fresh grouped rollouts with synchronous updates and writes exact trajectories before
training consumes them. Both require `--tasks` and `--dpo-adapter`.

`nanopt pipeline run` executes the calibrated Base-to-GRPO recipe as hash-linked child runs. Use
`--run-id ID --resume` to verify and skip completed stages; retained failures are retried as a new
numbered attempt rather than overwritten.

`nanopt agent run` evaluates a scripted oracle or the pinned Qwen policy in resettable MiniSWE
tasks. `--backend docker` is the secure reference path; `--backend fake` is only for trusted local
fixtures and CPU labs. The model never supplies test commands or arbitrary shell text. Capping tasks
or turns labels the result non-representative. v0.1 records exact model response token IDs but never
trains or updates the policy in this environment.

`nanopt report build RUN_DIR` is offline and model-free. It rebuilds all headline aggregates from
`samples.jsonl`. Run `uv run nanopt COMMAND --help` for the complete typed option surface.

The CLI does not advertise placeholders that silently do nothing.
