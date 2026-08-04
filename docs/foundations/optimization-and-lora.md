# Optimization and LoRA without unnecessary abstraction

## Learning objectives

After this chapter, you should be able to:

- trace a scalar loss through gradients, accumulation, clipping, and one optimizer step;
- explain why frozen base weights and trainable LoRA parameters have different roles;
- estimate which tensors dominate training memory;
- distinguish an optimizer step from a minibatch or epoch;
- inspect exactly which parameters changed.

## One update, explicitly

For parameters $\theta$ and scalar loss $L(\theta)$, plain gradient descent is

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t),
$$

where $\eta$ is the learning rate. PyTorch separates the phases: compute the loss, call
`backward()`, optionally accumulate/clip gradients, call `optimizer.step()`, then clear gradients.
NanoPT keeps that order visible in each trainer rather than hiding it in callbacks.

Gradient accumulation divides a logical batch into microbatches. If $K$ microbatch losses are
averaged, each backward call contributes $L_k/K$ before one optimizer step. An epoch is a pass over
the scheduled examples; it is not synonymous with a step.

## LoRA constrains the update space

For a frozen weight matrix $W \in \mathbb{R}^{d_{out}\times d_{in}}$, LoRA learns

$$
W' = W + \frac{\alpha}{r}BA,
$$

with $A \in \mathbb{R}^{r\times d_{in}}$, $B \in \mathbb{R}^{d_{out}\times r}$, rank $r$, and
scale $\alpha$. The base matrix remains frozen; optimizers receive only adapter parameters. This
reduces trainable parameter and optimizer-state memory, not all memory. Base weights, activations,
attention intermediates, gradients for adapters, and temporary logits still exist.

[`configure_lora`](https://github.com/shenli/nanopt/blob/main/src/nanopt/models/adapters.py)
prints and validates the trainable scope. The SFT, DPO, and GRPO entry points each spell out their
own forward/backward/update order because their data and objectives differ.

## Memory vocabulary

- **Weights:** model and adapter parameters.
- **Optimizer state:** for AdamW, moment estimates for trainable parameters.
- **Gradients:** accumulated derivatives for trainable parameters.
- **Activations:** forward values retained for backward; sequence length is important.
- **Reserved CUDA memory:** allocator reservation, not identical to live tensors.

BF16 halves many model/activation bytes relative to FP32, while numerically sensitive reductions
remain FP32. Activation checkpointing trades additional forward computation for fewer retained
activations.

## CPU lab

```bash
uv run python labs/19_optimizer_step.py
```

The lab freezes one scalar, optimizes another, and asserts the exact update scope. The real LoRA
lifecycle is covered by `tests/unit/models/test_adapters.py` and the supervised fine-tuning GPU
validation report.

## Common mistakes

- Clearing gradients before the last accumulated microbatch.
- Dividing a loss twice when accumulation is already normalized.
- Passing frozen base parameters to the optimizer.
- Calling BF16 model compute “all-BF16 training” while reductions remain FP32.
- Treating peak reserved memory as a portable promise for every GPU with the same capacity.

## At scale and further reading

NanoPT omits sharded optimizer state, tensor/pipeline parallelism, and quantized training in v0.1.
Read the [LoRA paper](https://arxiv.org/abs/2106.09685) for the low-rank method and the
[PEFT documentation](https://huggingface.co/docs/peft/) for the adapter infrastructure NanoPT uses.

## Exercises

1. Calculate the LoRA parameter count for a `4096 × 4096` matrix at rank 16.
2. Explain which memory categories LoRA does and does not eliminate.
3. Add a second trainable scalar to lab 19 and predict both gradients before running it.
