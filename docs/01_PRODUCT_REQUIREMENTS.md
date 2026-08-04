# Product Requirements

## 1. Product objective

NanoPT must let a learner run and inspect a complete small-model post-training pipeline without requiring cluster infrastructure. The repository must be useful in three modes:

1. **Course mode:** follow chapters and labs in order.
2. **Code-reading mode:** inspect minimal implementations and tests.
3. **Experiment mode:** change data, rewards, group size, sampling, or loss normalization and compare runs.

## 2. Primary user journey

A new user should be able to execute the following progression:

```bash
uv sync --frozen --extra dev --extra docs
uv run nanopt doctor
uv run nanopt config resolve --recipe math_pipeline --stage base_eval
uv run nanopt calibrate --mode load --device cuda
uv run nanopt pipeline run --recipe math_pipeline --tasks <tasks.jsonl> --device cuda
uv run nanopt report build artifacts/runs/<run-id>
```

The exact CLI can evolve, but the user experience must preserve these concepts: environment diagnosis, explicit resolved configuration, short calibration, formal run, and local report.

## 3. Functional requirements

### FR-1: Environment diagnosis

`nanopt doctor` must report, in machine-readable and human-readable form:

- OS and architecture;
- Python version;
- PyTorch version;
- CUDA runtime and driver information;
- visible GPU names and count;
- total and currently free VRAM;
- BF16 and TF32 capability checks;
- required and optional dependency status;
- Hugging Face cache location;
- Docker availability for agent labs;
- whether the detected hardware matches a known profile.

The command must not download a model or mutate the environment.

### FR-2: Typed, reproducible configuration

Configuration resolution must:

- combine one hardware profile, one model profile, and one experiment profile;
- validate all fields with typed models;
- reject unknown keys by default;
- show provenance for every resolved value;
- write `resolved_config.yaml` before a run begins;
- support explicit CLI overrides;
- avoid implicit “magic” adaptation that makes a run irreproducible.

### FR-3: Calibration

Before a long GPU run, `nanopt calibrate` must execute a representative short step and record:

- peak allocated and reserved VRAM;
- tokens per second for forward/backward where applicable;
- generation tokens per second for rollout experiments;
- wall-clock time;
- sequence lengths and batch shapes;
- whether the configured soft and hard VRAM budgets were exceeded.

Calibration may recommend changes. It must not silently change the resolved training configuration.

### FR-4: Data generation and validation

The reference task generator must create versioned JSONL datasets with deterministic seeds and explicit split definitions. It must include:

- SFT examples;
- preference prompts with chosen and rejected responses;
- RL prompts and exact ground-truth answers;
- held-out, compositional, and adversarial-format evaluation splits;
- dataset cards with generation parameters and leakage checks.

### FR-5: Baseline evaluation

Before training, the pipeline must evaluate the base model with exactly the same generation and parsing code used for later checkpoints. It must save every prompt, exact sampled token IDs where applicable, decoded response, parser result, reward components, and evaluation metric.

### FR-6: SFT

The SFT implementation must support:

- completion-only cross-entropy;
- explicit prompt and completion masks;
- LoRA adapters;
- gradient accumulation;
- BF16 on the reference GPU;
- gradient checkpointing as a configurable option;
- deterministic checkpointing and resume;
- evaluation during and after training;
- adapter-only checkpoint export.

### FR-7: DPO

The DPO implementation must support:

- chosen/rejected pairs;
- policy and reference sequence log probabilities;
- precomputed reference log probabilities to save memory;
- standard DPO loss and preference accuracy;
- configurable beta;
- completion-only masks;
- LoRA training;
- comparison against the SFT checkpoint.

### FR-8: GRPO/RLVR

The reference GRPO implementation must support:

- synchronous on-policy rollouts;
- multiple completions per prompt;
- exact sampled token IDs and generation-time log probabilities;
- deterministic verifier rewards;
- group-relative advantage calculation;
- old-policy ratio and clipping;
- configurable loss normalization;
- optional KL/reference regularization;
- reward, entropy, length, clipping, KL, and group-degeneracy metrics;
- inspectable trajectory JSONL.

### FR-9: Evaluation and reporting

The report must compare Base, SFT, DPO, and GRPO checkpoints on fixed datasets. At minimum it must show:

- parse rate;
- exact-answer pass@1;
- pass@k where sampled evaluation is used;
- average and distribution of response length;
- reward component means;
- preference accuracy for DPO;
- entropy proxy;
- KL or log-probability drift where defined;
- per-task-family performance;
- regression examples;
- reward-hacking or parser-failure examples;
- peak VRAM, throughput, and wall-clock time for each stage.

### FR-10: Agent environment

The MiniSWE environment must provide:

- versioned task definitions;
- deterministic initial snapshots;
- resettable workspaces;
- allow-listed structured tools;
- public and hidden tests;
- tool/time/token budgets;
- terminal conditions;
- structured trajectories;
- deterministic final-state verification;
- isolation from the host and hidden verifier.

v0.1 required rollout and evaluation, not policy optimization in this environment. v0.2 added
exact-token Agent SFT, v0.3 added Mini Agent RL, and v0.4 adds only a deterministic systems
simulation around longer rollout control.

### FR-11: Documentation

The documentation site must include:

- a conceptual map of post-training;
- mathematical foundations with correctly rendered formulas;
- implementation walkthroughs tied to source files and tests;
- labs with expected outputs and troubleshooting;
- paper and code-reading guides;
- explanations of what the single-GPU implementation simplifies or omits;
- an industrial systems section covering rollout/training separation, stale data, KV cache, sandboxes, and evaluation gates.

## 4. Non-functional requirements

### NFR-1: Readability

Core algorithm files should normally remain below roughly 500 lines and should avoid framework-style inversion of control. Line count is a guide, not a target.

### NFR-2: Reproducibility

Every run records:

- git commit and dirty status;
- resolved model revision;
- dataset fingerprints;
- seeds;
- complete resolved configuration;
- package versions;
- hardware details;
- start/end timestamps;
- checkpoint lineage.

### NFR-3: Offline continuation

After model and datasets have been downloaded once, training and evaluation should support an offline mode that fails clearly when a required artifact is absent.

### NFR-4: Security

Agent tasks must run with no network access by default, no GPU access, a non-root user, filesystem and process limits, and a hard timeout. Tools must validate paths and disallow traversal outside the workspace.

### NFR-5: Testability

Mathematical primitives must be testable on CPU with tiny tensors. Integration tests must use a tiny randomly initialized model created locally or a cached fixture and must not require network access.

### NFR-6: Portability

The first validated platform is Linux with CUDA. Code should avoid unnecessary vendor-specific assumptions, but unvalidated platforms must be clearly labeled instead of claimed as supported.

### NFR-7: Honest performance communication

README tables must distinguish proposed settings from measured results. Runtime and memory numbers must link to a run manifest in the same commit or release.

## 5. Success metrics

The project is successful when:

- a learner can explain each training stage by tracing one example through the code;
- the reference pipeline completes on the supported GPU without manual source edits;
- held-out evaluation demonstrates measurable stage-by-stage behavior changes;
- failed and degenerate runs remain interpretable through saved trajectories and metrics;
- documentation builds without malformed formulas;
- contributors can add a new hardware profile through a documented evidence process.
