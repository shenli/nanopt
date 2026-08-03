# Model loading, rendering, and LoRA lifecycle

## Learning objectives

After this lesson, you should be able to:

- explain why model and tokenizer revisions must be recorded separately;
- describe how NanoPT proves a prompt/completion token boundary;
- explain why decoded-text boundary searches are unsafe;
- identify which model parameters a LoRA adapter changes;
- trace an adapter through create, clone, freeze, select, save, and load operations.

## Reproducible Qwen loading

The reference starting checkpoint is `Qwen/Qwen3-0.6B-Base`. A model ID alone is not immutable: a
repository branch can point to different content later. The canonical profile pins both inputs to
one Hub commit, and the
[`load_qwen3_base`](https://github.com/shenli/nanopt/blob/main/src/nanopt/models/loading.py) contract
therefore records resolved model and tokenizer commit hashes. It also makes dtype, safetensors,
remote-code trust, local-only behavior, and device mapping explicit.

The loader is the only M2 component allowed to contact the Hugging Face Hub. Normal CPU tests replace
the auto loaders and cannot use the network. A separate network/GPU smoke check will exercise the
real reference checkpoint when compatible hardware is deliberately selected.

## Token boundaries come from rendering

For supervised data, NanoPT renders two sequences:

1. prompt messages plus the assistant generation marker;
2. the same messages plus the trusted assistant completion.

The tokenizer is asked for PyTorch tensors explicitly so this contract does not depend on whether a
Transformers version defaults to Python lists, tokenizer `Encoding` objects, or a `BatchEncoding`.
The first token sequence must be an exact prefix of the second. Its length is therefore the first
completion-token index. [`ChatRenderer`](https://github.com/shenli/nanopt/blob/main/src/nanopt/models/renderer.py)
constructs the action mask directly from this integer boundary.

NanoPT never decodes the full sequence and searches for the prompt string. Decoding may normalize
spaces, special tokens, or byte-level representations, so a text offset is not a reliable token
offset. If a chat template does not preserve the exact prefix, the renderer rejects it.

The template text is hashed and recorded. Thinking mode is also an explicit renderer option because
changing it can change tokens and invalidate cached sequence log probabilities.

## LoRA adapters as named state

LoRA leaves base weights immutable and learns low-rank update matrices on selected modules. Before
injection, NanoPT maps every requested target suffix such as `q_proj` to concrete model module names.
A missing target is an error rather than a silent no-op.

The lifecycle helpers in
[`adapters.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/models/adapters.py) support:

- attaching a new named adapter;
- counting total and trainable parameters;
- cloning weights into a new stage adapter;
- freezing a reference adapter;
- temporarily selecting an adapter and restoring the previous selection;
- saving one adapter with safetensors;
- loading it onto an independently constructed base model.

The save/load test constructs a tiny Qwen3 model locally, modifies its LoRA state, saves only the
adapter, reconstructs the same base weights, and verifies matching logits. No model is downloaded.

## Common mistakes

- Recording only a human-readable model ID, not resolved revisions.
- Allowing `trust_remote_code=True` without an explicit security decision.
- Inferring completion boundaries from decoded strings.
- Updating base weights accidentally alongside LoRA parameters.
- Training the frozen DPO or GRPO reference adapter.
- Saving every adapter when an artifact claims to contain only one stage.

See the offline boundary and lifecycle fixtures under
[`tests/unit/models`](https://github.com/shenli/nanopt/tree/main/tests/unit/models).
