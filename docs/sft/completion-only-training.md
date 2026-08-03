# Completion-only supervised fine-tuning

## Learning objectives

After this chapter, you should be able to explain which tokens SFT optimizes, inspect an exact
prompt/completion boundary, follow one optimizer step without a trainer framework, and distinguish
teacher-forced loss improvement from generated-answer quality.

## Why completion-only loss exists

An instruction record contains context supplied to the model and a response we want it to imitate.
If the loss also scores the user prompt, training spends capacity copying text the model was given.
NanoPT therefore optimizes only assistant completion tokens while retaining prompt tokens as causal
context.

For token log probabilities $\ell_{i,t}$ and a binary completion mask $m_{i,t}$, the objective is

$$
\mathcal{L}_{\mathrm{SFT}} =
-\frac{\sum_{i,t} m_{i,t}\ell_{i,t}}{\sum_{i,t}m_{i,t}}.
$$

The denominator is the number of active completion tokens—not the padded sequence length and not
the number of examples. This detail matters when micro-batches contain different response lengths.

## Token coordinates and the causal shift

Suppose a rendered record is:

```text
token position       0     1     2     3
role                 prompt prompt reply reply
action mask          0     0     1     1
logits row predicts  1     2     3     -
```

The action mask describes token positions. Causal logits row 1 is located in the prompt but predicts
the first response token at position 2, so it must receive gradient. “Prompt-only tokens have zero
loss” does not mean detaching the prompt: its hidden states still provide context for the response.

[`ChatRenderer`](https://github.com/shenli/nanopt/blob/main/src/nanopt/models/renderer.py) renders the
prompt and full conversation separately and requires the former to be an exact token prefix. It
never searches decoded strings for a boundary. [`CompletionOnlyCollator`](https://github.com/shenli/nanopt/blob/main/src/nanopt/sft/data.py)
right-pads IDs, attention masks, and action masks together; it rejects overlong examples instead of
silently truncating a target.

## The readable training step

[`train_sft`](https://github.com/shenli/nanopt/blob/main/src/nanopt/sft/trainer.py) makes the control
flow explicit:

1. Build a deterministic shuffled batch schedule.
2. Group micro-batches into one optimizer boundary.
3. Weight each micro-batch loss by its active-token count.
4. Backpropagate and clip the global LoRA gradient norm.
5. Apply AdamW and the visible warmup-plus-cosine learning rate.
6. Record metrics, validate, and checkpoint only after the optimizer step.

Only parameters whose names identify LoRA tensors may be trainable. This turns an accidental
unfrozen base layer into an error rather than an expensive surprise.

## Accumulation and resume semantics

Naively dividing every micro-batch loss by the configured accumulation count gives short responses
too much weight. NanoPT first counts active tokens across the optimizer group and scales each
micro-batch by

$$
w_j = \frac{\text{active tokens in micro-batch }j}
{\text{active tokens in the optimizer group}}.
$$

The deterministic schedule is regenerated on resume, and completed optimizer groups are skipped.
Adapter weights, AdamW state, RNG state, step counts, and hashes are stored together. Mid-accumulation
checkpoints are deliberately unsupported because their partially accumulated gradients are easy to
misinterpret.

## Metrics and their limits

Inspect completion NLL, completion-token accuracy, gradient norm/clipping, learning rate, active
tokens, tokens per second, and peak CUDA memory. Validation NLL answers “does the policy assign more
probability to trusted completions?” It does not answer “does greedy generation follow the answer
format or solve held-out arithmetic?” The protected generation evaluator must establish that
separately.

## Run the CPU lab

```bash
uv run python labs/07_completion_only_sft.py
```

The lab uses a four-token record with mask `[0, 0, 1, 1]`, trains a tiny transition table, and checks
that completion NLL decreases. The hand-checkable tests in
[`tests/unit/sft`](https://github.com/shenli/nanopt/tree/main/tests/unit/sft) additionally prove
padding invariance, prompt-target gradient masking, adapter/optimizer checkpoint integrity, and
clean-boundary resume equivalence.

## Run the real stage

After generating the dataset, calibrate one non-representative optimizer step before the full run:

```bash
uv run nanopt calibrate --mode sft --tasks artifacts/data/tasks.jsonl --device cuda
uv run nanopt train sft --tasks artifacts/data/tasks.jsonl --device cuda
```

Then evaluate the saved adapter using `nanopt eval run --adapter ADAPTER_DIR`. Keep the calibration
label separate from representative evidence.

## Common mistakes

- Shifting the action mask twice drops the first completion token.
- Marking padding active changes the loss when another example length changes.
- Averaging micro-batch means equally changes the token-level objective.
- Treating lower teacher-forced NLL as generation success hides format failures.
- Saving in the middle of accumulation makes resume semantics ambiguous.
- Loading an adapter without recording its content hash breaks checkpoint lineage.

At larger scale, systems add distributed sharding, fused optimizers, packing, asynchronous
checkpoint uploads, and fault-tolerant dataloaders. NanoPT omits those so the mathematical and state
boundaries remain visible on one GPU.
