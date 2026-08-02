# Tokens, masks, and causal log probabilities

## Learning objectives

After this lesson, you should be able to:

- explain why causal logits and target token IDs are shifted by one position;
- distinguish full-token coordinates from causal-prediction coordinates;
- construct a mask that excludes prompts, padding, and tokens after EOS;
- calculate a completion sequence log probability by hand;
- identify the error caused by shifting a mask twice or not shifting it at all.

## Why this topic comes first

SFT, DPO, PPO, and GRPO all need the probability assigned to particular completion tokens. A
one-position error can make an implementation optimize prompt tokens, ignore the first completion
token, or assign a probability to the wrong target. The training loop may still produce a finite
loss, so unit tests must establish the coordinate convention before model training begins.

NanoPT uses one rule everywhere:

> A mask is first constructed in full-token coordinates. It is shifted exactly once when it is
> combined with causal token log probabilities.

## The causal shift

Suppose a batch of token IDs has shape `[B, T]` and model logits have shape `[B, T, V]`, where `V`
is the vocabulary size. Logits at position $t$ predict the token at position $t+1$:

$$
\text{prediction\_logits} = \text{logits}[:, :-1, :]
$$

$$
\text{target\_ids} = \text{input\_ids}[:, 1:]
$$

Both now have a causal prediction length of $T-1$. The first input token has no log probability in
the returned tensor because there is no earlier logits position that predicts it. The final logits
position is unused because its next token is not present in `input_ids`.

| Quantity | Shape | Coordinate meaning |
|---|---:|---|
| `logits` | `[B, T, V]` | distribution produced at every input position |
| `input_ids` | `[B, T]` | full input-token positions |
| `action_mask` | `[B, T]` | whether each full input token is an optimized action |
| `token_logps` | `[B, T - 1]` | probability of token `j + 1` predicted at position `j` |
| `prediction_mask` | `[B, T - 1]` | shifted action status for token `j + 1` |

The canonical implementation is
[`causal_token_logps`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/logprobs.py).
It performs `log_softmax` in FP32, even if incoming model logits use BF16 or FP16.

## Building a completion mask

Consider this padded sequence:

```text
position       0         1          2       3       4        5
token        <bos>    Compute      2+2      4     <eos>    <pad>
region       prompt    prompt     prompt  completion completion padding
action_mask    0         0          0       1       1        0
```

Here EOS is included as a generated action. If a renderer chooses not to optimize EOS, the value at
position 4 is zero instead. [`completion_action_mask`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/masks.py)
also makes every token after the first terminal token inactive, even if an attention mask happens to
mark it as non-padding.

After the causal shift:

```text
predicted token position     1      2      3      4      5
prediction_mask              0      0      1      1      0
```

The mask still refers to tokens 3 and 4. Only its tensor index has changed to match the logits row
that predicts each token.

## A calculation by hand

Assume the model assigns probability $0.5$ to token `4` and probability $0.8$ to the following EOS.
The completion sequence log probability is the sum over active tokens:

$$
\log \pi(y \mid x) = \log(0.5) + \log(0.8) = \log(0.4) \approx -0.9163.
$$

Sequence probabilities multiply, while sequence log probabilities add. DPO in NanoPT uses this
sum convention; it does not silently length-normalize the value.

Run the CPU lab from the repository root:

```bash
uv run python labs/00_tokens_and_masks.py
```

The lab prints both masks, all selected causal token log probabilities, and the expected and actual
sequence result. It does not download a tokenizer or model.

## Masked reductions

[`masked_sum` and `masked_mean`](https://github.com/shenli/nanopt/blob/main/src/nanopt/core/reductions.py)
convert values and masks to FP32 before reduction. `masked_mean` raises an error if any reduced slice
has zero active values. Adding a small epsilon would keep the program running, but it would hide an
invalid training example and produce a misleading loss.

The hand-computable fixtures live beside the code:

- [`test_masks.py`](https://github.com/shenli/nanopt/blob/main/tests/unit/core/test_masks.py)
- [`test_logprobs.py`](https://github.com/shenli/nanopt/blob/main/tests/unit/core/test_logprobs.py)
- [`test_reductions.py`](https://github.com/shenli/nanopt/blob/main/tests/unit/core/test_reductions.py)

## Common mistakes

### Using `action_mask[:, :-1]`

This assigns the action status of token $t$ to the distribution predicting token $t+1$. NanoPT uses
`action_mask[:, 1:]` because the selected log probability belongs to the target token.

### Masking only padding

An attention mask distinguishes real tokens from padding, but it does not distinguish prompt tokens
from completion tokens. SFT and policy-gradient objectives require both boundaries.

### Searching decoded text for the prompt boundary

Decoded text can normalize whitespace or special tokens. The renderer must preserve token
boundaries directly and pass the first completion-token index to the mask builder.

### Reducing in model precision

Thousands of BF16 log probabilities can accumulate avoidable error. NanoPT selects token log
probabilities and performs reductions in FP32 while still allowing gradients to flow to the model.

## Exercises

1. Change the lab to exclude EOS. Which prediction-mask entry changes, and what is the new sequence
   log probability?
2. Add right padding after EOS and verify that the result remains unchanged.
3. Create a two-sequence batch with different completion lengths. Compare a global token mean with
   the mean of the two sequence means.
4. Explain why `input_ids[:, 0]` cannot have a teacher-forced causal log probability in this API.
