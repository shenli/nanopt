# Exact generation and inspectable evaluation

## Learning objectives

After this chapter, you should be able to:

- trace one sampled token ID and its stopping decision without `transformers.generate`;
- distinguish policy and behavior log probabilities;
- separate parse rate, exact correctness, pass@k, and uncertainty intervals;
- explain why examples are written before aggregate metrics;
- rebuild a report without loading a model.

Milestone 3 connects the token probabilities from the foundations chapters to complete model
responses. NanoPT uses a small token-at-a-time sampler instead of delegating the reference path to
`transformers.generate`. The slower implementation keeps every probability and stopping decision
visible.

## Coordinates returned by the sampler

For a prompt of length $P$ and a generated completion of length $C$,
`sample_autoregressive` returns one-dimensional sequences:

| Field | Shape | Meaning |
| --- | --- | --- |
| `prompt_token_ids` | $[P]$ | Exact rendered prompt IDs |
| `generated_token_ids` | $[C]$ | IDs chosen by the sampler |
| `active_mask` | $[C]$ | Generated actions, including the first EOS |
| `policy_logps` | $[C]$ | Log probabilities under the unmodified model softmax |
| `behavior_logps` | $[C]$ | Log probabilities after temperature and top-p filtering |

At generation step $j$, the model receives all $P+j$ tokens already known. The final logits row
predicts the next token. NanoPT converts that row to FP32 before `log_softmax`, selects one ID, saves
the corresponding log probability, and appends the ID to the prefix.

The reference rollout distribution uses temperature 1 and top-p 1. Under that setting,

$$
\log \pi_{\text{policy}}(a_j \mid s_j)
=
\log \pi_{\text{behavior}}(a_j \mid s_j).
$$

Evaluation may use other settings. In that case both values are saved conceptually because
temperature or nucleus renormalization changes the behavior distribution. Later policy-gradient
code must not pretend those distributions are identical.

In deterministic mode, `behavior_logps` copy the raw-policy diagnostic. Greedy choice is not
presented as a stochastic behavior distribution.

## Greedy versus sampled evaluation

Deterministic evaluation chooses the largest raw-policy log probability at every step. It uses one
completion per task and does not consume the random generator.

Sampled evaluation constructs a private `torch.Generator` for each task/sample pair. The seed is a
SHA-256-derived function of:

```text
base evaluation seed + task ID + sample index
```

This avoids dependence on task iteration order and global PyTorch RNG state. Repeating the same
plan produces the same seed schedule under the documented backend and software environment.

## EOS and stopping masks

`first_eos_mask` marks only the first EOS token in generated-token coordinates.
`active_through_eos_mask` includes that EOS by default and masks every later position. For IDs
`[4, 2, 9, 2]` with EOS ID 2, the masks are:

```text
first EOS:          [0, 1, 0, 0]
active through EOS: [1, 1, 0, 0]
```

The sampler stops immediately after the first EOS or an explicitly configured exact token stop
sequence, so its returned mask is all true. Stop-sequence tokens remain in the exact trajectory.
The standalone mask functions also handle padded batches and imported trajectories that contain
positions after EOS.

Reports separate generic EOS, task-protocol stop, and length-limit fractions. A model that reaches
the exact `</answer>` stop boundary should not be presented as a zero-termination run merely because
it did not emit the tokenizer's generic EOS token.

## Teacher-forced parity

The most important sampler test joins the prompt and exact sampled IDs, scores the full sequence in
one teacher-forced model pass, and compares the causal token log probabilities with the values
captured during generation. With temperature 1 and top-p 1, the sampled-token values must match.
This detects one-token shifts, decoding/re-tokenization mistakes, and probability calculations from
the wrong distribution.

Run the CPU lab:

```bash
uv run python labs/06_exact_generation.py
```

## Example-first evaluation

`evaluate_to_artifacts` depends on a small `EvaluationBackend` protocol, not a checkpoint class.
The same loop can therefore evaluate Base, SFT, DPO, or GRPO as long as the backend can render a
prompt, generate exact IDs, and decode those IDs for the parser.

The runner enforces the profile's maximum rendered prompt length and fails instead of silently
truncating. Truncation changes task meaning and must become an explicit versioned policy before it
is allowed.

For every completion, NanoPT appends one typed record to `samples.jsonl` before updating the final
aggregate. The record contains:

- run, checkpoint, task, split, sample, and seed identity;
- generation-configuration fingerprint;
- exact prompt and completion token IDs;
- response text and finish reason;
- parser and verifier outcomes;
- generation time and compact diagnostics.

An interruption therefore leaves completed examples available for inspection. Aggregates can be
rebuilt without another model run.

The parser-facing decode skips tokenizer control tokens such as EOS. The exact EOS ID remains in
`completion_token_ids`; only the inspection text drops it. Evaluation never decodes and then
re-tokenizes a trajectory.

## pass@k and intervals

For $n$ samples with $c$ correct and $n \ge k$, NanoPT reports the standard estimator

$$
\operatorname{pass@k}
=
1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

For $n=4$, $c=2$, and $k=2$, the result is $1-1/6=5/6$. The implementation evaluates the ratio as
a small product rather than constructing large combinations.

Accuracy-like rates include a labeled 95% Wilson interval. The pass@k headline is the estimator
above; its accompanying interval is explicitly labeled as a Wilson interval over the direct event
that at least one of the first $k$ samples passed. These quantities are not silently conflated.

## Build a local baseline run

First create the deterministic versioned task artifact:

```bash
uv run nanopt data generate
```

`eval run` requires the sibling `dataset_manifest.json`. Before model loading, it verifies every
split count and canonical task hash. The run manifest records the dataset fingerprint plus checksums
for both the task JSONL and split manifest.

On the reference machine, calibrate the real load and then a deliberately limited evaluation:

```bash
uv run nanopt calibrate --mode load --device cuda
uv run nanopt calibrate \
  --mode eval \
  --tasks artifacts/data/arithmetic_v1/tasks.jsonl \
  --device cuda
```

The evaluation calibration is marked non-representative. A full deterministic baseline uses:

```bash
uv run nanopt eval run \
  --tasks artifacts/data/arithmetic_v1/tasks.jsonl \
  --mode deterministic \
  --checkpoint-id base \
  --device cuda
```

Run sampled evaluation separately with `--mode sampled`; never merge deterministic and sampled
records in one JSONL file.

For the pinned base-model GPU smoke, use the checked-in orchestration and evidence validator
instead of copying individual commands. The script retains its historical milestone identifier so
old evidence remains reproducible:

```bash
bash scripts/run_m3_reference_smoke.sh
```

## Reports are safe to share

`nanopt report build RUN_DIR` rebuilds `summary.json`, `report.md`, and a self-contained
`report.html`. Reports render only whitelisted aggregates and safe run/checkpoint labels. They use
relative links and do not embed environment details, prompts, response bodies, absolute paths, or
secret-like text. Full responses remain in `samples.jsonl` for deliberate inspection.

This is a protection against accidental leakage, not a claim that arbitrary run artifacts are safe
to publish. Review `samples.jsonl` and the run manifest before sharing a run directory.
