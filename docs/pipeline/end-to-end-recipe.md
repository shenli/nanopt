# End-to-end recipe and resumable lineage

## Learning goals

After this chapter you should be able to:

- explain why Base, SFT, DPO, and GRPO remain separate processes;
- follow a checkpoint hash through training and evaluation;
- distinguish a logical stage from one of its retry attempts;
- resume a stopped pipeline without silently trusting stale artifacts;
- rebuild the final comparison from saved child runs.

## The visible stage sequence

The official recipe deliberately spells out calibration, training, evaluation, and reporting:

```text
load calibration → evaluation calibration → Base evaluation
→ SFT calibration → SFT → SFT evaluation
→ controlled preferences
→ DPO calibration → DPO → DPO evaluation
→ GRPO calibration → GRPO → GRPO evaluation → repeated GRPO evaluation
→ comparison report
```

This is more verbose than calling three trainers from one Python function, but the boundaries are
useful. A completed child has its own resolved config, environment record, metrics, samples, report,
and run manifest. The parent pipeline manifest records the child's manifest hash and the input and
output checkpoint hashes.

## Run the recipe

First create the deterministic task file:

```bash
uv run nanopt data generate \
  --output artifacts/tmp/m7/data/tasks.jsonl \
  --manifest artifacts/tmp/m7/data/dataset_manifest.json
```

Then run the official recipe:

```bash
uv run nanopt pipeline run \
  --tasks artifacts/tmp/m7/data/tasks.jsonl \
  --recipe math_pipeline \
  --artifacts-root artifacts/tmp/m7/pipelines \
  --run-id learning-run \
  --device cuda
```

Use `--local-files-only` after the pinned model is present in the Hugging Face cache. The complete
reference protocol, including a fresh environment from `uv.lock`, is:

```bash
bash scripts/run_m7_reference_pipeline.sh
```

The reference command is intentionally local. NanoPT does not require or use GitHub Actions for
this gate.

## Read the parent before the children

Open `pipeline_manifest.json` first. Each stage has four ideas worth checking:

| Field | Question it answers |
| --- | --- |
| `input_checkpoint_sha256` | Which exact policy entered this stage? |
| `output_checkpoint_sha256` | Which exact policy or artifact left it? |
| `attempts` | Did this stage fail or retry, and how long did each attempt take? |
| `child_manifest_sha256` | Has the child manifest changed since the parent accepted it? |

For example, the SFT adapter hash must equal the DPO input hash. The DPO adapter hash must equal the
GRPO input hash. Evaluation stages also bind to those same hashes, so a report cannot accidentally
compare a different adapter with a familiar label.

## Resume is verification, not continuation by assumption

If the process stops, rerun with the same paths and add `--resume`:

```bash
uv run nanopt pipeline run \
  --tasks artifacts/tmp/m7/data/tasks.jsonl \
  --recipe math_pipeline \
  --artifacts-root artifacts/tmp/m7/pipelines \
  --run-id learning-run \
  --resume \
  --local-files-only \
  --device cuda
```

Before skipping a completed stage, NanoPT re-hashes its child manifest and output checkpoint. A
hash mismatch stops the resume rather than guessing which file is authoritative. A failed stage is
retained and the next attempt receives a suffix such as `dpo-retry-2`; the failure remains in the
parent disclosure log.

The unit test `tests/unit/pipeline/test_pipeline_runner.py` is the shortest executable explanation of these
rules. It demonstrates both a verified skip and a controlled failure followed by a retained retry.

## The protected final comparison

Base, SFT, DPO, and GRPO use the same deterministic evaluator and frozen test splits. The report
shows exact-answer accuracy, a 95% Wilson interval, parse rate, and checkpoint SHA-256. It does not
use protected results to tune the already-frozen recipe.

The final GRPO evaluation is repeated. Run IDs and measured generation seconds naturally differ,
so reproducibility compares the meaningful generation evidence: task/sample identity, prompt and
completion token IDs, decoded response, parser state, verifier state, and finish reason. A mismatch
is a release failure.

## Exercises

1. Trace the value of `stages.sft.output_checkpoint_sha256` into the DPO stage and SFT evaluation.
2. Explain why changing a completed adapter file must stop resume instead of creating a retry.
3. Run `tests/unit/pipeline/test_pipeline_runner.py`, then add a second output file and decide how it should
   participate in the stage hash.
4. Rebuild `report.md` from `comparison.json` and verify that no model load is required.

## Industrial mapping

NanoPT runs one stage at a time on one GPU. A production orchestrator may schedule containers on a
cluster and store artifacts in object storage, but the safety contract is the same: immutable input
identity, explicit child state, atomic parent updates, retained retries, measured resource use, and
a report derived from frozen evidence.
