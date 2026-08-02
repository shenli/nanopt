# Technical Architecture

## 1. Architectural goal

The architecture must keep the post-training control flow visible while centralizing only the pieces whose correctness must be shared across algorithms: token rendering, masks, log-probability calculations, configuration, artifacts, evaluation, and checkpoint lineage.

NanoPT should not grow a generic `Trainer` base class in v0.1. SFT, DPO, and GRPO should each have an explicit entry point and loop.

## 2. Layered view

```text
CLI and recipes
    ↓
configuration resolution and run lifecycle
    ↓
algorithm vertical slices: SFT / DPO / GRPO
    ↓
shared mathematical primitives and model adapters
    ↓
data, rollout, verifier, evaluation, artifacts
    ↓
PyTorch + Transformers + PEFT
```

The agent environment is a sibling subsystem rather than a special case of the static-prompt trainer:

```text
task registry
→ sandbox reset
→ observation
→ structured tool action
→ state transition
→ trajectory
→ hidden final-state verifier
```

## 3. Major modules

### 3.1 Configuration and run lifecycle

Responsibilities:

- load and validate hardware, model, and experiment profiles;
- resolve overrides deterministically;
- create a unique run directory;
- capture environment and source-control metadata;
- enforce calibration and support-state rules;
- write metrics and artifacts atomically;
- support clean resume from checkpoints.

Key rule: a run must never depend on an unrecorded default.

### 3.2 Model and tokenizer integration

Responsibilities:

- load the exact model revision and tokenizer revision;
- validate required tokenizer fields and special tokens;
- render prompts and completions while preserving token boundaries;
- inject LoRA modules with PEFT;
- create, copy, freeze, load, and save adapter states;
- expose a context manager for selecting a policy or reference adapter;
- record trainable and total parameter counts.

The initial integration targets `Qwen/Qwen3-0.6B-Base`. Model-specific assumptions must be isolated behind a small adapter/renderer interface rather than scattered through algorithms.

### 3.3 Core mathematical primitives

Required primitives include:

- causal token log probabilities;
- completion/action masks;
- masked sequence reductions;
- entropy estimates;
- KL estimators;
- DPO margins and loss;
- group-relative advantages;
- PPO-style probability ratios and clipping;
- reward normalization and degeneracy detection.

Every function documents tensor shapes, dtypes, masking semantics, and numerical stability behavior.

### 3.4 Data subsystem

Responsibilities:

- generate deterministic synthetic tasks;
- validate schemas and exact answers;
- assign leakage-safe splits;
- create SFT records, preference records, and RL prompt records;
- fingerprint files and generator versions;
- collate batches without losing prompt/completion boundaries;
- cache reference log probabilities with dataset and checkpoint fingerprints.

### 3.5 Rollout subsystem

The reference GRPO path uses an explicit synchronous autoregressive sampler rather than a serving engine. It must return:

- prompt token IDs;
- generated token IDs exactly as sampled;
- active-token mask;
- old policy log probabilities for sampled tokens;
- decoded text for inspection only;
- finish reason;
- token counts and timing;
- reward components after verification.

Training must consume the exact stored token IDs. Decoding followed by re-tokenization is forbidden because it can change token boundaries and invalidate log probabilities.

For the reference training recipe, sampling uses temperature `1.0`, `top_p=1.0`, and no top-k truncation. This keeps the behavior distribution equal to the model softmax and makes the policy ratio unambiguous. More complex samplers may be added later with explicit behavior-policy accounting.

### 3.6 Evaluation subsystem

Evaluation must be checkpoint-agnostic and use the same renderer, generator, parser, and verifier across stages. It writes example-level results before aggregating metrics.

### 3.7 Reporting subsystem

Reports are generated locally from run artifacts. No hosted telemetry service is required. A report is reproducible from files in the run directory and contains links to individual examples and trajectories.

### 3.8 Agent environment subsystem

The agent subsystem defines:

- task and snapshot schemas;
- a typed observation/action protocol;
- allow-listed tools;
- a Docker-backed sandbox implementation;
- a local fake sandbox for unit tests;
- public test feedback;
- a separate hidden verifier workspace;
- budgets and terminal conditions;
- exact trajectory serialization.

## 4. Checkpoint lineage

The official pipeline is sequential:

```text
Qwen3-0.6B-Base
    ↓ SFT
SFT LoRA adapter
    ↓ DPO, initialized from SFT
DPO LoRA adapter
    ↓ GRPO, initialized from DPO
GRPO LoRA adapter
```

The base weights remain immutable and are not duplicated in run artifacts.

### 4.1 SFT

- load immutable base;
- create a new trainable SFT adapter;
- save adapter, tokenizer metadata, config, and lineage.

### 4.2 DPO

- load base plus SFT adapter;
- compute and cache frozen SFT reference log probabilities for every preference record;
- clone the SFT adapter into a trainable DPO adapter;
- continue optimization using cached reference values;
- save DPO adapter with the SFT parent checkpoint ID.

Reference-log-probability cache keys must include model revision, tokenizer revision, renderer version, SFT adapter hash, dataset fingerprint, truncation policy, and reduction convention.

### 4.3 GRPO

- load base plus DPO adapter as the initial policy;
- clone the DPO adapter into a trainable GRPO policy adapter;
- preserve a frozen DPO adapter when KL/reference evaluation is enabled;
- generate rollouts with the policy adapter;
- store old log probabilities at rollout time;
- update the policy with PPO-style clipping and group-relative advantages.

Because GRPO data is generated online, reference log probabilities cannot be globally precomputed. If `kl_beta > 0`, compute them after rollout in a no-gradient pass by switching to the frozen reference adapter, then switch back to the policy adapter. The initial low-memory recipe may set `kl_beta: 0.0`, but this choice must be explicit and the report must still measure checkpoint drift on evaluation data.

## 5. Exact token and mask conventions

Given full token IDs `input_ids` with shape `[B, T]`, logits at position `t` predict token `t + 1`:

```python
prediction_logits = logits[:, :-1, :]
target_ids = input_ids[:, 1:]
```

Per-token log probabilities have shape `[B, T - 1]`. A label/action mask must be shifted to the same positions.

The project must define one canonical convention:

- `input_ids[:, 0]` has no predicted-token log probability;
- `token_logps[:, j]` is the log probability of `input_ids[:, j + 1]`;
- `action_mask[:, j]` indicates whether `input_ids[:, j + 1]` is an optimized completion/action token;
- padding, prompt tokens, and tokens after the first terminal token have mask value zero;
- EOS may be included as an action token when generated or present in the target, controlled by one documented setting.

All algorithms use these primitives. They must not independently reimplement shifting rules.

## 6. Run artifact contract

Every run directory has the following minimum structure:

```text
artifacts/runs/<run-id>/
├── resolved_config.yaml
├── run_manifest.json
├── environment.json
├── metrics.jsonl
├── events.jsonl
├── samples.jsonl
├── checkpoints/
├── cache/
├── plots/
├── report.md
└── report.html
```

GRPO adds `trajectories.jsonl`. Agent rollouts add an `agent_trajectories/` directory and sandbox/verifier summaries.

Files should be append-safe or atomically replaced. A partial or interrupted run must remain inspectable.

## 7. Dependency policy

### Required runtime dependencies

- Python 3.11 as the reference interpreter;
- PyTorch;
- Transformers;
- PEFT;
- safetensors;
- PyYAML;
- Pydantic;
- Typer or argparse for CLI;
- Jinja2;
- NumPy;
- tqdm;
- matplotlib for static reports.

### Development dependencies

- pytest;
- pytest-cov;
- Ruff;
- mypy or pyright;
- MkDocs Material;
- a MathJax-compatible docs plugin/configuration;
- pre-commit.

### Optional extras

- `trl`: parity examples only;
- `quant`: bitsandbytes experiments after the BF16 reference path;
- `docs`: documentation build;
- `dev`: tests and static analysis.

Do not add an optional dependency to the default environment merely because a future milestone may use it.

## 8. Failure handling

- Configuration errors fail before model download or GPU allocation.
- Dataset-schema failures identify the record and field.
- NaN/Inf detection captures the current batch, recent metrics, and checkpoint before stopping.
- OOM errors emit current shapes and a deterministic list of suggested parameter reductions.
- Verifier exceptions yield a distinct `verifier_error`, not an incorrect-answer reward.
- Parser failures are reported separately from wrong answers.
- Unsupported hardware is a warning for smoke tests and a hard error for `--require-validated-profile`.

## 9. Extension boundaries

Future distributed or accelerated implementations should conform to the same schemas and mathematical primitives but live behind separate backends. The single-GPU reference loop must remain available and readable even after faster backends exist.
