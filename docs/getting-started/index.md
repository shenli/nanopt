# Getting Started

This chapter takes you from a fresh checkout to the first runnable NanoPT lab. It also introduces
the small amount of Python, tensor, probability, and autograd vocabulary used throughout the
course. You do not need prior experience with reinforcement learning, RLHF, DPO, GRPO, or Qwen.

## Who this course is for

NanoPT is written for software engineers, ML practitioners, and technically experienced learners
who want to understand post-training without treating trainer APIs as magic. You should be
comfortable reading Python functions and using a terminal.

You should recognize functions, lists, dictionaries, loops, exceptions, type annotations, and
context managers. Advanced metaprogramming is not required; NanoPT deliberately keeps algorithmic
control flow explicit.

## Learning objectives

After this chapter, you should be able to:

- install the locked NanoPT environment and run the readiness lab;
- read tensor shapes such as `[batch, sequence, vocabulary]`;
- distinguish token IDs, masks, logits, losses, and gradients;
- explain why probabilities multiply while log probabilities add;
- resolve a typed experiment configuration and inspect its provenance;
- use `nanopt doctor` to distinguish CPU readiness from validated GPU support.

## 1. Install NanoPT

NanoPT requires Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), Git, and a terminal. Clone the
repository and create the locked environment:

```bash
git clone https://github.com/shenli/nanopt.git
cd nanopt
uv sync --frozen --extra dev --extra docs
uv run nanopt --help
```

Run the CPU-only readiness lab:

```bash
uv run python labs/00_prerequisites.py
```

The environment sync may download packages. The lab itself does not use the network, download a
model, or require a GPU.

### Why the locked environment matters

On macOS and other non-reference platforms, PyTorch comes from PyPI. On Linux x86-64, `uv` selects
the pinned CUDA-enabled PyTorch wheel from PyTorch's package index. Installing a CUDA wheel does
not prove that the local driver or GPU is compatible; the diagnosis step below checks the actual
machine. [ADR-0004](../adr/0004-reference-pytorch-wheel.md) records the dependency decision and its
trade-offs.

Tests marked `gpu`, `network`, or `reference` are not prerequisites for the normal CPU learning
path.

## 2. Learn the notation used by the course

### Tensor shapes

A tensor is an array with a dtype, device, and shape. NanoPT names dimensions rather than relying
on their sizes:

| Symbol | Meaning | Example |
|---|---|---|
| `B` | batch size | two prompts processed together |
| `T` | sequence length | tokens in each padded sequence |
| `V` | vocabulary size | one score for every possible next token |
| `G` | rollout group size | completions sampled for one GRPO prompt |

Logits with shape `[B, T, V]` contain one vocabulary-sized score vector for every token position in
every batch item. Token IDs with shape `[B, T]` select which vocabulary entries occurred.

Important PyTorch operations used throughout the course include:

- `tensor[:, 1:]`: keep every batch item and drop the first sequence position;
- `unsqueeze(-1)`: add a size-one dimension;
- broadcasting: combine compatible shapes without manually copying data;
- `gather`: select entries using integer indices;
- `sum` and `mean`: reduce one or more dimensions.

The readiness lab demonstrates each operation with tiny tensors.

### Dtypes and devices

| Dtype | Purpose |
|---|---|
| `torch.int64` | token IDs and integer indices |
| `torch.bool` | attention and action masks |
| `torch.float32` | log probabilities, loss reductions, and stored rollout values |
| `torch.bfloat16` | model computation on supported GPUs |

A CPU tensor cannot be combined directly with a CUDA tensor. NanoPT validates device alignment at
subsystem boundaries so a mistake fails with a specific message.

### Probability and logarithms

A model produces unconstrained logits $z_v$. Softmax converts them into a probability distribution:

$$
p_v = \frac{\exp(z_v)}{\sum_j \exp(z_j)}.
$$

Language-model sequence probabilities multiply conditional token probabilities:

$$
p(y_1, y_2 \mid x) = p(y_1 \mid x)\,p(y_2 \mid x, y_1).
$$

Taking the logarithm turns that product into a sum:

$$
\log p(y_1, y_2 \mid x)
=
\log p(y_1 \mid x) + \log p(y_2 \mid x, y_1).
$$

Probabilities below one have negative log probabilities, and a more likely event has a less
negative log probability.

### Losses, gradients, and autograd

A loss is a scalar that measures the behavior an optimizer should improve. A gradient reports how
the loss changes when a parameter changes. For

$$
L(w) = (w - 3)^2,
$$

the derivative is $2(w-3)$. At $w=1$, the gradient is $-4$. The readiness lab asks PyTorch to
calculate that exact result. Later chapters apply the same mechanism to model parameters and use
LoRA to restrict which parameters are trainable.

You should understand this four-step flow:

1. the forward pass produces logits and a scalar loss;
2. the backward pass produces gradients;
3. the optimizer uses those gradients to update trainable parameters;
4. masks determine which token losses contribute.

### Language-model vocabulary

- **Token:** a vocabulary unit represented by an integer ID.
- **Tokenizer:** code that maps text to token IDs and back.
- **Prompt:** input context supplied before the optimized or generated completion.
- **Completion:** target or generated tokens after the prompt boundary.
- **Logit:** an unnormalized score for a possible next token.
- **Causal model:** a model whose output at position $t$ predicts position $t+1$.
- **Padding:** placeholder positions used to give sequences a common batch length.
- **Attention mask:** marks real tokens versus padding.
- **Action mask:** marks completion tokens that participate in an objective.

## 3. Resolve a configuration

NanoPT validates every hardware, model, experiment, and recipe profile with strict typed models.
Unknown fields fail before a model is downloaded. Resolve the reference GRPO configuration:

```bash
uv run nanopt config resolve \
  --hardware rtx_4070_ti_super_16gb \
  --model qwen3_0_6b_base \
  --experiment math_grpo \
  --set rollout.group_size=2 \
  --output resolved_config.yaml
```

This writes `resolved_config.yaml` and `resolved_config.provenance.yaml`. The first file contains the
effective configuration. The second records where each value came from.

An unprefixed dotted override targets the experiment profile. Use `hardware.` or `model.` prefixes
for those namespaces. Only scalar leaves may be overridden; unknown paths, list replacement through
the CLI, and type mismatches are rejected.

## 4. Diagnose the machine

`nanopt doctor` is read-only. It reports the operating system, Python and PyTorch versions,
dependencies, CUDA runtime, visible GPUs, VRAM, compute capability, numerical capabilities, cache
location, Docker state, and hardware-profile match.

```bash
uv run nanopt doctor --json doctor.json
```

Exit codes are part of the command contract:

| Code | Meaning |
|---:|---|
| 0 | The requested validated profile matches and required capabilities are usable. |
| 2 | CUDA is usable, but the profile is unvalidated, unspecified, or does not match in non-strict mode. |
| 3 | A required dependency or usable CUDA device is missing. |
| 4 | The requested profile does not match under `--strict-profile`. |

The JSON report validates against `specs/schemas/doctor_report.schema.json`. A CPU-only learner may
receive exit code 3 because CUDA is unavailable; that does not prevent the CPU chapters and labs
from running.

## Hardware expectations

The readiness lab, core mathematics, synthetic-data tools, and normal unit tests run on CPU. Model
loading, calibration, and reference training require the documented GPU checks. Passing this page
does not prove that a training recipe fits in GPU memory.

## Readiness checklist

You are ready for the first foundation chapter if you can answer these questions:

1. What does each dimension in `[B, T, V]` represent?
2. Why are token IDs integers while logits are floating point?
3. Why does adding log probabilities correspond to multiplying probabilities?
4. What does `loss.backward()` calculate?
5. Why are an attention mask and an action mask not interchangeable?
6. What is the difference between resolving a hardware profile and validating the actual machine?

The checklist is diagnostic, not an admission test. Continue when the examples make sense and
return here whenever you need a refresher.

## Next step

Continue to [Tokens, masks, and causal log probabilities](../foundations/tokens-masks-logprobs.md),
then run:

```bash
uv run python labs/01_tokens_and_masks.py
```
