# Prerequisites and readiness check

## Who this course is for

NanoPT is written for software engineers, ML practitioners, and technically experienced learners
who want to understand post-training without treating trainer APIs as magic. You do not need prior
experience with reinforcement learning, RLHF, DPO, GRPO, distributed training, or the internals of
Qwen. Those topics are taught as part of the course.

You should be comfortable reading Python functions and using a terminal. This chapter introduces
the specific mathematical and PyTorch vocabulary used by the first lessons and tells you where to
pause for additional study.

## Learning objectives

After this chapter, you should be able to:

- run commands from the repository root with `uv`;
- read tensor shapes such as `[batch, sequence, vocabulary]`;
- distinguish integer token IDs, boolean masks, and floating-point logits;
- explain why probabilities multiply while log probabilities add;
- recognize softmax, a scalar loss, and a gradient;
- identify which course steps are CPU-only and which eventually require a GPU.

## Development tools

NanoPT requires:

- Python 3.11 or 3.12;
- `uv` for the locked Python environment;
- Git for version control;
- a terminal opened at the repository root.

Install the project environment before running the readiness lab:

```bash
uv sync --extra dev --extra docs
uv run python labs/00_prerequisites.py
```

The first command may download packages. The lab itself is CPU-only, does not use the network, and
does not download a model.

## Python knowledge

You should recognize:

- functions, arguments, and return values;
- lists, dictionaries, tuples, and dataclasses;
- loops and conditional statements;
- exceptions such as `ValueError`;
- type annotations such as `list[str]` and `Tensor | None`;
- context managers written with `with`.

You do not need advanced metaprogramming. NanoPT deliberately avoids hiding algorithmic flow behind
callbacks and factories.

## Reading tensor shapes

A tensor is an array with a dtype, device, and shape. NanoPT names dimensions rather than relying on
their sizes:

| Symbol | Meaning | Example |
|---|---|---|
| `B` | batch size | two prompts processed together |
| `T` | sequence length | tokens in each padded sequence |
| `V` | vocabulary size | one score for every possible next token |
| `G` | rollout group size | completions sampled for one GRPO prompt |

For example, logits with shape `[B, T, V]` contain one vocabulary-sized score vector for every
token position in every batch item. Token IDs with shape `[B, T]` select which vocabulary entries
actually occurred.

Important PyTorch operations used throughout the course include:

- `tensor[:, 1:]`: keep all batch items and drop the first sequence position;
- `unsqueeze(-1)`: add a size-one dimension;
- broadcasting: combine compatible shapes without manually copying data;
- `gather`: select entries using integer indices;
- `sum` and `mean`: reduce one or more dimensions.

The readiness lab demonstrates each idea with tiny tensors. You do not need a GPU to learn tensor
coordinates.

## Dtypes and devices

NanoPT commonly uses:

| Dtype | Purpose |
|---|---|
| `torch.int64` | token IDs and integer indices |
| `torch.bool` | attention and action masks |
| `torch.float32` | log probabilities, loss reductions, and stored rollout values |
| `torch.bfloat16` | model computation on supported GPUs |

A tensor on the CPU cannot be combined directly with a tensor on a CUDA GPU. NanoPT validates
device alignment at subsystem boundaries so a mistake fails with a specific message.

## Probability and logarithms

A discrete probability is between zero and one, and the probabilities of all mutually exclusive
choices sum to one. A model produces unconstrained logits $z_v$. Softmax converts them into a
distribution:

$$
p_v = \frac{\exp(z_v)}{\sum_j \exp(z_j)}.
$$

Language-model sequence probabilities multiply conditional token probabilities:

$$
p(y_1, y_2 \mid x) = p(y_1 \mid x)\,p(y_2 \mid x, y_1).
$$

Products of many small probabilities become inconvenient numerically. Taking the logarithm turns
the product into a sum:

$$
\log p(y_1, y_2 \mid x)
=
\log p(y_1 \mid x) + \log p(y_2 \mid x, y_1).
$$

You should be comfortable with the facts that `log(1) = 0`, probabilities below one have negative
log probabilities, and a more likely event has a less negative log probability.

## Losses, gradients, and autograd

A loss is a scalar that measures the behavior an optimizer should improve. A gradient reports how
the loss changes when a parameter changes. PyTorch records differentiable operations when a tensor
has `requires_grad=True`; calling `loss.backward()` populates parameter gradients.

For the small example

$$
L(w) = (w - 3)^2,
$$

the derivative is $2(w-3)$. At $w=1$, the gradient is $-4$. The readiness lab asks PyTorch to
calculate that exact result. Later chapters explain how the same mechanism applies to millions of
model parameters and why NanoPT restricts trainable updates with LoRA.

You do not need to derive every neural-network gradient by hand. You do need to understand that:

- the forward pass produces logits and a loss;
- the backward pass produces gradients;
- the optimizer uses those gradients to update trainable parameters;
- masks determine which token losses are allowed to contribute.

## Causal language-model vocabulary

- **Token:** a vocabulary unit represented by an integer ID.
- **Tokenizer:** code that maps text to token IDs and back.
- **Prompt:** input context supplied before the optimized or generated completion.
- **Completion:** target or generated tokens after the prompt boundary.
- **Logit:** an unnormalized score for a possible next token.
- **Causal model:** a model whose output at position $t$ predicts position $t+1$.
- **Padding:** placeholder positions used to give sequences a common batch length.
- **Attention mask:** marks real tokens versus padding.
- **Action mask:** marks completion tokens that participate in an objective.

The next lesson turns these definitions into exact tensor coordinates and tested code.

## Hardware expectations

The prerequisites, core-math labs, synthetic-data tools, and normal unit tests run on CPU. Model
loading, calibration, and reference training arrive in later milestones and require explicitly
documented GPU checks. Passing this chapter does not validate the proposed RTX 4070 Ti SUPER
profile or prove that a training recipe fits in 16 GB.

## Readiness checklist

You are ready for the token and mask chapter if you can answer these questions:

1. What does each dimension in `[B, T, V]` represent?
2. Why are token IDs integers while logits are floating point?
3. Why does adding log probabilities correspond to multiplying probabilities?
4. What does `loss.backward()` calculate?
5. Why are an attention mask and an action mask not interchangeable?

If the self-check passes but one answer is unclear, continue with the next chapter and return here
when needed. The checklist is diagnostic, not an admission test.

## Next step

Continue to [Tokens, masks, and causal log probabilities](../foundations/tokens-masks-logprobs.md),
then run:

```bash
uv run python labs/01_tokens_and_masks.py
```
