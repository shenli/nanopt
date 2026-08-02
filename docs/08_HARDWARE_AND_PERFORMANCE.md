# Hardware and Performance

## 1. Reference platform

NanoPT v0.1 targets one platform for its first validation:

```text
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
VRAM: 16 GB GDDR6X
GPU count: 1
CUDA compute capability: 8.9
OS: Linux x86_64
```

The public NVIDIA specification lists 16 GB memory and compute capability 8.9 for the RTX 4070 Ti SUPER. The project must still detect the actual board, driver, runtime, available VRAM, and BF16 support at execution time.

Reference: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/

## 2. Support-state model

Hardware profiles have one of these states:

- `proposed_unvalidated`: config exists but no full evidence bundle has passed;
- `smoke_tested`: environment and short calibration passed;
- `validated`: every required reference stage and report passed on the named profile;
- `deprecated`: profile is retained for history but no longer maintained.

Only `validated` profiles appear in README support claims.

## 3. Initial memory strategy

The proposed reference model has approximately 0.6B parameters. The initial path uses:

- BF16 frozen base weights;
- BF16 or FP32 LoRA parameters according to PEFT behavior and measured stability;
- AdamW over adapter parameters only;
- SDPA attention;
- gradient checkpointing enabled for training by default;
- no 4-bit quantization;
- no 8-bit optimizer requirement;
- no vLLM;
- no `torch.compile` in the validated baseline.

This is intentionally simpler than QLoRA. Quantization should be added only after the unquantized adapter path is measured, because quantization introduces extra dependencies and changes the learning/debugging surface.

## 4. Proposed safety budgets

Initial profile values:

```yaml
soft_peak_reserved_gib: 14.5
hard_peak_reserved_gib: 15.2
minimum_free_vram_before_start_gib: 14.0
```

These values are proposals, not measurements. The final profile must reflect observed behavior on the owner's machine, including display usage and allocator fragmentation.

- Exceeding the soft budget produces a warning and suggested reductions.
- Exceeding the hard budget fails calibration.
- A successful calibration does not guarantee a long run; a full reference run is required for validation.

## 5. Calibration protocol

Calibration has stage-specific modes.

### 5.1 Model load

Measure:

- peak VRAM during load;
- steady reserved/allocated memory;
- CPU RAM;
- load time;
- resolved model revision.

### 5.2 Evaluation forward

Use representative maximum prompt/completion lengths and measure forward latency, token throughput, and peak memory.

### 5.3 SFT step

Execute at least:

- one warmup forward/backward;
- one measured optimizer step with the intended accumulation and sequence lengths;
- gradient checkpointing and adapter settings identical to the full run.

### 5.4 DPO step

Include chosen and rejected sequences, cached reference values, and the intended concatenation strategy.

### 5.5 GRPO iteration

Include:

- grouped generation;
- reward parsing/verification;
- current-policy scoring;
- optional reference scoring if enabled;
- backward and optimizer step;
- retained rollout tensors.

Because generation and training peaks can occur at different times, log phase-specific peaks as well as whole-process peaks.

## 6. Memory instrumentation

Record at minimum:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
torch.cuda.mem_get_info()
```

Reset peak statistics only at documented phase boundaries. Also capture a sanitized `nvidia-smi` snapshot when available. Explain the difference between allocated, reserved, total, and free memory in the course.

## 7. Performance instrumentation

Required timing buckets:

- model load;
- data loading/collation;
- forward;
- backward;
- optimizer step;
- rollout generation;
- verifier;
- evaluation;
- checkpoint save;
- report build.

Synchronize CUDA before wall-clock measurements where necessary. Do not synchronize every inner-loop operation in normal runs because it distorts throughput.

Report:

- active training tokens per second;
- generated tokens per second;
- prompts/completions per second;
- optimizer steps per hour;
- total stage wall-clock time;
- rollout/training utilization split for GRPO.

## 8. Proposed starting configurations

These are conservative starting points for calibration, not promised final values.

### SFT

```yaml
max_sequence_length: 512
micro_batch_size: 4
gradient_accumulation_steps: 8
lora_rank: 16
gradient_checkpointing: true
```

### DPO

```yaml
max_prompt_length: 256
max_completion_length: 256
micro_batch_size_pairs: 2
gradient_accumulation_steps: 8
reference_mode: precomputed
```

### GRPO

```yaml
prompt_batch_size: 1
group_size: 4
max_prompt_length: 256
max_completion_length: 128
update_epochs: 1
gradient_accumulation_steps: 8
kl_beta: 0.0
```

The coding agent must not tune silently around these values. Calibration recommendations are printed; the owner or reference-run process commits the chosen config.

## 9. OOM response policy

On OOM, report:

- stage and phase;
- input shapes and active token counts;
- group size and number of retained rollouts;
- allocated/reserved/free VRAM immediately before failure if available;
- whether gradient checkpointing and reference scoring were active;
- a deterministic recommendation order.

Suggested reduction order by stage:

### SFT

1. microbatch size;
2. max sequence length;
3. LoRA target breadth or rank;
4. enable/verify gradient checkpointing;
5. only then consider quantization.

### DPO

1. pair microbatch size;
2. max completion length;
3. max prompt length;
4. concatenate-vs-separate forward strategy;
5. LoRA rank;
6. quantization as an optional backend.

### GRPO

1. max completion length;
2. prompt batch size;
3. group size, but never below two;
4. update minibatch size;
5. disable optional reference KL;
6. LoRA rank;
7. optional quantization/accelerated backends later.

Do not automatically change group size or sequence length during a run; this changes the experiment.

## 10. Reference evidence bundle

A validated hardware recipe must include:

```text
reference_runs/<release>/<hardware-id>/
├── doctor.json
├── environment.json
├── resolved_configs/
├── calibration/
├── stage_manifests/
├── metrics/
├── final_report.html
├── final_report.md
├── checksums.txt
└── VALIDATION.md
```

`VALIDATION.md` lists exact commands, commit, driver, CUDA/PyTorch versions, total wall time, phase peaks, failures/retries, and any deviations from the proposed plan.

## 11. Adding another GPU

A contributor must provide:

1. a new hardware profile;
2. `nanopt doctor` output;
3. calibration evidence for every required stage;
4. a full pipeline run or a documented reduced support level;
5. peak-memory and throughput records;
6. CI/schema validation;
7. a support-table update that accurately states what passed.

A GPU with the same nominal VRAM is not automatically compatible: architecture, BF16 behavior, driver, memory bandwidth, and desktop reservation can change results.

## 12. Runtime claims

Do not publish estimated training times as measured values. The first full pipeline run will establish official wall-clock ranges. Subsequent releases should show the distribution across repeated runs when possible.
