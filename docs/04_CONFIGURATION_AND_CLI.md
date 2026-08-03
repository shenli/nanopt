# Configuration and CLI

## 1. Configuration model

NanoPT separates three orthogonal concerns:

1. **Hardware profile:** verified capabilities and safety limits.
2. **Model profile:** model identity, tokenizer, precision, renderer, and adapter targets.
3. **Experiment profile:** data, algorithm, rollout, optimizer, evaluation, and logging parameters.

A recipe sequences experiments and defines checkpoint lineage. This separation allows future hardware support without placing GPU-name branches inside algorithm code.

## 2. Resolution rules

Resolution order, lowest to highest precedence:

```text
package defaults
< hardware profile
< model profile
< experiment profile
< recipe stage overrides
< explicit CLI overrides
```

Every resolved field must retain provenance. Unknown fields are errors. Ambiguous list merging is forbidden: lists replace lists unless a specific field defines another behavior.

The resolver writes a stable, sorted `resolved_config.yaml` and a provenance map before starting work.

## 3. Hardware profile schema

Important fields:

```yaml
schema_version: 1
id: rtx_4070_ti_super_16gb
support_status: validated
platform:
  os: linux
  architecture: x86_64
accelerator:
  vendor: nvidia
  count: 1
  name_regex: "RTX 4070 Ti SUPER"
  total_vram_gib: 16
  compute_capability: "8.9"
precision:
  preferred_compute_dtype: bfloat16
  allow_tf32: true
memory_budget:
  soft_peak_reserved_gib: 14.5
  hard_peak_reserved_gib: 15.2
runtime:
  attention_backend: sdpa
  torch_compile: false
  gradient_checkpointing_default: true
validation:
  evidence_manifest: docs/reference/evidence/m7-reference-pipeline-92564f3.json
```

The hard threshold is a safety limit, while the M7 evidence records the measured 7.41 GiB peak.
`support_status` is `validated` only because `evidence_manifest` points to committed reviewed
evidence for the pinned reference path.

## 4. Model profile schema

```yaml
schema_version: 1
id: qwen3_0_6b_base
source:
  provider: huggingface
  model_id: Qwen/Qwen3-0.6B-Base
  revision: null
  trust_remote_code: false
loading:
  dtype: bfloat16
  low_cpu_mem_usage: true
renderer:
  type: tokenizer_chat_template
  enable_thinking: false
adapter:
  method: lora
  rank: 16
  alpha: 32
  dropout: 0.0
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

A null revision means “resolve once, record the immutable revision in the run manifest.” Official reference recipes should later pin the validated revision.

The Instruct sibling may be used for isolated agent-environment debugging, but it is not the source checkpoint for the official Base → SFT → DPO → GRPO learning path.

## 5. Experiment profiles

Each experiment profile includes:

- dataset identifiers and split;
- renderer and truncation policy;
- algorithm-specific parameters;
- optimizer and schedule;
- effective batch accounting;
- evaluation cadence;
- checkpoint cadence;
- metrics and sample logging;
- seed and determinism mode.

Every config must expose the quantities learners need to reason about. Avoid a single opaque `training_args` dictionary.

## 6. CLI surface

Recommended initial commands:

```text
nanopt doctor
nanopt config resolve
nanopt calibrate
nanopt data generate
nanopt data validate
nanopt eval run
nanopt train sft
nanopt train dpo
nanopt train grpo
nanopt pipeline run
nanopt agent run
nanopt report build
nanopt artifacts inspect
```

### 6.1 `nanopt doctor`

Example:

```bash
nanopt doctor --json artifacts/doctor.json
```

Exit codes:

- `0`: required environment is usable;
- `2`: usable but hardware is unvalidated or differs from requested profile;
- `3`: missing required dependency or no usable CUDA device;
- `4`: profile mismatch under strict mode.

### 6.2 `nanopt config resolve`

```bash
nanopt config resolve \
  --hardware rtx_4070_ti_super_16gb \
  --model qwen3_0_6b_base \
  --experiment math_grpo \
  --set rollout.group_size=4 \
  --output resolved_config.yaml
```

The command prints a concise table of values and provenance, then writes the complete file.

### 6.3 `nanopt calibrate`

```bash
nanopt calibrate --config resolved_config.yaml --steps 2
```

Calibration runs the same model/data/loss path as the full experiment. It must not substitute a smaller model or shorter sequence unless the user explicitly requests a calibration override and the report labels it non-representative.

### 6.4 Training commands

```bash
nanopt train sft --config configs/resolved/math_sft.yaml
nanopt train dpo --config configs/resolved/math_dpo.yaml
nanopt train grpo --config configs/resolved/math_grpo.yaml
```

Each command creates a run ID unless `--run-dir` is supplied for resume.

### 6.5 Pipeline command

```bash
nanopt pipeline run \
  --tasks artifacts/data/arithmetic_v1/tasks.jsonl \
  --recipe math_pipeline \
  --artifacts-root artifacts/pipelines \
  --run-id reference
```

The recipe is a sequence of independently resumable stages. It must not collapse all stages into one process. Each stage gets its own run manifest and the pipeline gets a parent manifest.

Resume verifies the saved child-manifest and checkpoint hashes before skipping work:

```bash
nanopt pipeline run \
  --tasks artifacts/data/arithmetic_v1/tasks.jsonl \
  --recipe math_pipeline \
  --artifacts-root artifacts/pipelines \
  --run-id reference \
  --resume
```

### 6.6 Agent command

```bash
nanopt agent run \
  --recipe mini_swe \
  --checkpoint artifacts/checkpoints/grpo \
  --task-split smoke
```

The command checks Docker isolation requirements before loading the model.

## 7. CLI overrides

Support dotted scalar overrides only in v0.1. Examples:

```bash
--set training.max_steps=50
--set rollout.group_size=2
--set evaluation.num_samples=100
```

Reject unknown paths and type mismatches. Print all overrides in the manifest. Do not support arbitrary Python expressions.

## 8. Secrets and tokens

Hugging Face authentication may be read from the standard library/cache mechanism. NanoPT must never serialize access tokens into resolved configs, manifests, logs, or reports. The reference model is public, so the golden path should not require a token unless rate or infrastructure policy changes.

## 9. Run IDs and paths

Recommended run ID:

```text
YYYYMMDD-HHMMSS_<stage>_<short-config-hash>
```

Do not include user names, host names, prompts, or secrets in paths. A run manifest may record a sanitized host identifier if explicitly enabled.

## 10. Resume semantics

Resume must validate:

- algorithm and stage;
- model/tokenizer revision;
- adapter lineage;
- dataset fingerprint;
- optimizer/scheduler compatibility;
- critical configuration fields.

Safe logging/evaluation changes may be allowed. Any override accepted during resume must be recorded. A `--force-resume` escape hatch should be omitted from v0.1 or should produce a visibly tainted run status.
