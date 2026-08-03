# Cached-reference DPO training

## Learning objectives

After this chapter, you should be able to:

- trace one preference pair from token rendering to its DPO loss;
- explain why the trainable policy must start as an exact SFT copy;
- list every input that invalidates the frozen-reference cache;
- distinguish policy preference accuracy from implicit-reward accuracy;
- inspect a DPO run without relying on a trainer framework.

## Two roles begin at the same checkpoint

The frozen reference policy and the initial trainable policy are both the M4 SFT adapter. Their
initial margins therefore match and each pair begins with loss $log 2$. NanoPT first scores every
pair under the frozen SFT adapter, then clones that adapter under the name `dpo` and makes only the
clone trainable.

For pair batch size $B$, the rendered chosen and rejected tensors each contain:

```text
input_ids:      [B, sequence]
attention_mask: [B, sequence]
action_mask:    [B, sequence]
reference_logp: [B]
```

The action mask is false on prompt and padding tokens. It is shifted exactly once by
[`completion_sequence_logps`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/logprobs.py),
which returns an FP32 **sum** for each completion.

## The cache is part of the objective contract

Reference scoring is expensive but does not change during DPO. The single-GPU path precomputes it.
[`ReferenceCacheIdentity`](https://github.com/shenli/nanopt/blob/main/src/nanopt/dpo/cache.py) binds:

- model and tokenizer revisions;
- the content hash of the SFT adapter;
- renderer version and chat-template hash;
- preference-dataset fingerprint;
- prompt and completion limits;
- reject-rather-than-truncate policy;
- chat-terminator inclusion policy;
- sequence-sum reduction.
- concatenated or separate chosen/rejected forward layout, because BF16 kernels may not be bitwise
  identical across batch shapes.

Loading fails if any identity field or cache file hash changes. Before optimization, a deterministic
sample is scored live again and compared to the cache. A cache is therefore not a loose speed hack;
it is a fingerprinted dataset of model-derived values.

## Read the loop top to bottom

[`train_dpo`](https://github.com/shenli/nanopt/blob/main/src/nanopt/dpo/trainer.py) deliberately keeps
the algorithm visible:

1. materialize deterministic optimizer groups;
2. set the visible warmup-plus-cosine learning rate;
3. collate chosen/rejected tokens and cached reference scores;
4. run one concatenated policy forward;
5. compute both masked sequence sums;
6. call the hand-tested DPO objective;
7. weight micro-batches by pair count;
8. detect non-finite gradients, clip, and update LoRA parameters;
9. emit margins, accuracies, lengths, learning rate, gradients, and memory.

For policy margin $Delta_\theta$ and cached reference margin $Delta_{\mathrm{ref}}$, the optimized
quantity is

$$
L = -\log \sigma\left(\beta(\Delta_\theta - \Delta_{\mathrm{ref}})\right).
$$

`preference_accuracy` asks whether the policy itself assigns a positive chosen-minus-rejected
margin. `reward_accuracy` asks whether DPO has improved that margin relative to the frozen
reference. The latter is exactly 0 at initialization because the adapters are identical.

## Run the stages

```bash
uv run nanopt calibrate --mode dpo \
  --preferences artifacts/data/arithmetic_preferences_v1/preferences.jsonl \
  --sft-adapter artifacts/runs/sft/adapter/sft \
  --local-files-only --device cuda

uv run nanopt train dpo \
  --preferences artifacts/data/arithmetic_preferences_v1/preferences.jsonl \
  --sft-adapter artifacts/runs/sft/adapter/sft \
  --local-files-only --device cuda
```

Then evaluate the saved adapter with the same protected evaluator used for Base and SFT. A lower
held-out preference loss is not permission to hide a generation regression.

## Debugging checklist

- Initial loss differs from $log 2$: verify that the policy is an exact SFT clone.
- Cache/live values differ: compare every cache identity field before considering tolerance.
- Margins move backward: check chosen/rejected ordering and the sign in the objective.
- Loss changes with padding: inspect action masks and causal shifting.
- One rejection type wins: inspect `preference_breakdown.json` and token lengths.
- Protected accuracy falls: report the regression and change the recipe before freezing targets.

Large systems may score references on separate workers, pack sequences, or use distributed
trainers. Those optimizations must preserve the same masks, reductions, cache identity, and initial
policy/reference relationship demonstrated here.
