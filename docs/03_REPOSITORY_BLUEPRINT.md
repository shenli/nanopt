# Repository Blueprint

## Proposed public repository tree

```text
nanopt/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       ├── ci.yml
│       ├── docs.yml
│       └── gpu-reference.yml
├── .gitignore
├── .pre-commit-config.yaml
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── SECURITY.md
├── pyproject.toml
├── uv.lock
├── mkdocs.yml
│
├── configs/
│   ├── hardware/
│   │   └── rtx_4070_ti_super_16gb.yaml
│   ├── models/
│   │   ├── qwen3_0_6b_base.yaml
│   │   └── qwen3_0_6b_instruct.yaml
│   ├── experiments/
│   │   ├── base_eval.yaml
│   │   ├── math_sft.yaml
│   │   ├── math_dpo.yaml
│   │   ├── math_grpo.yaml
│   │   ├── ppo_toy.yaml
│   │   └── mini_swe_rollout.yaml
│   └── recipes/
│       ├── math_pipeline.yaml
│       └── mini_swe.yaml
│
├── docs/
│   ├── index.md
│   ├── getting-started/
│   ├── foundations/
│   ├── sft/
│   ├── preferences/
│   ├── dpo/
│   ├── policy-gradient/
│   ├── grpo-rlvr/
│   ├── agents/
│   ├── systems/
│   ├── labs/
│   ├── paper-guides/
│   └── reference/
│
├── src/nanopt/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── version.py
│   │
│   ├── config/
│   │   ├── models.py
│   │   ├── loader.py
│   │   ├── resolver.py
│   │   └── provenance.py
│   ├── runtime/
│   │   ├── doctor.py
│   │   ├── calibration.py
│   │   ├── environment.py
│   │   ├── run_context.py
│   │   ├── seeds.py
│   │   └── artifacts.py
│   ├── models/
│   │   ├── loading.py
│   │   ├── adapters.py
│   │   ├── checkpoint.py
│   │   ├── renderer.py
│   │   └── qwen.py
│   ├── core/
│   │   ├── logprobs.py
│   │   ├── masks.py
│   │   ├── reductions.py
│   │   ├── entropy.py
│   │   ├── kl.py
│   │   ├── dpo.py
│   │   ├── advantages.py
│   │   └── clipping.py
│   ├── data/
│   │   ├── schemas.py
│   │   ├── jsonl.py
│   │   ├── fingerprints.py
│   │   ├── collators.py
│   │   ├── arithmetic.py
│   │   ├── preferences.py
│   │   └── splits.py
│   ├── generation/
│   │   ├── sampler.py
│   │   ├── stopping.py
│   │   └── records.py
│   ├── train/
│   │   ├── common.py
│   │   ├── sft.py
│   │   ├── dpo.py
│   │   ├── grpo.py
│   │   └── ppo_toy.py
│   ├── eval/
│   │   ├── runner.py
│   │   ├── parser.py
│   │   ├── verifier.py
│   │   ├── metrics.py
│   │   ├── pass_at_k.py
│   │   └── regression.py
│   ├── reporting/
│   │   ├── build.py
│   │   ├── tables.py
│   │   ├── plots.py
│   │   └── templates/
│   └── agent/
│       ├── protocol.py
│       ├── tasks.py
│       ├── tools.py
│       ├── policies.py
│       ├── trajectory.py
│       ├── budgets.py
│       ├── sandbox/
│       │   ├── base.py
│       │   ├── local_fake.py
│       │   └── docker.py
│       └── verifier/
│           ├── public.py
│           └── hidden.py
│
├── tasks/
│   ├── arithmetic/
│   │   ├── README.md
│   │   └── generator_config.yaml
│   └── mini_swe/
│       ├── README.md
│       ├── registry.jsonl
│       ├── task_001/
│       └── ...
│
├── labs/
│   ├── 00_tokens_and_masks.py
│   ├── 01_logprob_by_hand.py
│   ├── 02_sft_vertical_slice.py
│   ├── 03_dpo_vertical_slice.py
│   ├── 04_grpo_advantages.py
│   ├── 05_reward_hacking.py
│   ├── 06_ppo_toy.py
│   ├── 07_agent_environment.py
│   └── 08_partial_rollout_simulation.py
│
├── scripts/
│   ├── smoke_cpu.sh
│   ├── smoke_gpu.sh
│   ├── reference_math_pipeline.sh
│   └── reference_mini_swe.sh
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── security/
│   ├── gpu/
│   └── fixtures/
│
└── artifacts/
    └── reference/
        └── .gitkeep
```

## File-level responsibilities

### `src/nanopt/core/`

Pure or near-pure tensor functions. No model download, CLI parsing, global logging configuration, or filesystem side effects. These functions are the mathematical heart of the course.

### `src/nanopt/train/`

Readable algorithm loops. Shared helpers may handle optimizer creation, checkpoint intervals, metric emission, and autocast context, but the loss construction and update order must remain visible in each file.

### `src/nanopt/generation/`

A small sampler with exact token/log-probability capture. Do not begin with a serving engine. The sampler should be easy to debug and slow enough to remain correct, then optimized only after profiling.

### `src/nanopt/models/`

Model-specific details, chat rendering, adapter lifecycle, and checkpoint metadata. Algorithms should depend on interfaces from this module, not hard-code Qwen layer names.

### `src/nanopt/eval/`

Parsing and verification are separate. Parsing converts text into a candidate structured answer; verification compares it with trusted task state. This distinction is essential for detecting parser failures and reward hacking.

### `src/nanopt/agent/`

Stateful environment mechanics. The model-facing public-test tool and the hidden release verifier must never share an implementation object or workspace.

## Public API policy

v0.1 does not promise a stable Python API. The CLI, artifact schemas, task schemas, and documented core functions should be treated as the first stability boundary. Mark experimental modules clearly.

## Code style

- type annotations for public functions;
- docstrings for non-obvious tensor semantics;
- assertions or validation at subsystem boundaries, not inside every inner-loop tensor operation;
- structured logging, not print statements in library code;
- no global mutable singleton configuration;
- no import-time CUDA initialization;
- no network access in tests unless explicitly marked;
- no hidden environment-variable behavior except documented cache/token conventions.

## Commit and pull-request structure

Prefer milestone-sized pull requests with one vertical slice and its tests. Avoid a single massive generated repository commit. Each PR description should include:

- scope and non-scope;
- architecture decisions;
- commands executed;
- test results;
- CPU/GPU status;
- artifact examples;
- remaining known limitations.
