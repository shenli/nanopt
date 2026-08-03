# NanoPT

**Nano Post-Training** is an executable course and white-box reference implementation for modern
language-model post-training. It is designed to make token masks, log probabilities, objectives,
rollouts, evaluation, and run lineage inspectable rather than hiding them behind a generic trainer.

The planned v0.1 learning path is:

```text
Qwen/Qwen3-0.6B-Base
→ completion-only LoRA SFT
→ controlled preference construction and DPO
→ synchronous GRPO/RLVR with a deterministic verifier
→ reproducible evaluation and regression reports
```

A separate MiniSWE subsystem teaches resettable, allow-listed, stateful coding-agent environments.
v0.1 evaluates that environment; it does not train an agent policy in it.

## Current status

NanoPT is pre-alpha. Milestone 1 provides the package, CLI, typed configuration resolution,
environment diagnosis, run-artifact lifecycle, documentation skeleton, and local CPU validation.
Milestone 2 completes the CPU-tested mathematical core, pinned Qwen loader and exact renderer, LoRA
adapter lifecycle, deterministic arithmetic generator and fingerprints, leakage-safe splits, and
strict parser/verifier. Milestone 3 completes exact autoregressive sampling, deterministic/sample
evaluation, pass@k and Wilson intervals, example-first JSONL artifacts, Markdown/HTML reports, and
load/eval calibration. Its real base-model smoke passed on the proposed reference GPU from a clean,
pinned checkout. Milestone 4 completes the readable completion-only LoRA SFT slice: protected parse
rate improved from 0% to 95.5% and exact-answer accuracy from 0% to 86.4% on the reference smoke.
Milestone 5 completes controlled preference construction and the white-box DPO stage. Its full
reference cache had zero live/cache error; held-out DPO loss improved from 0.693 to 0.626 while
protected exact accuracy moved from 86.4% to 84.1%, a disclosed 2.3-point regression.
Milestone 6 completes exact-token grouped RLVR and synchronous clipped GRPO. Its clean reference
run improved protected exact accuracy from DPO's 84.1% to 88.6%, with gains on compositional and
range splits, while storing all sampled IDs and old log probabilities before optimization.
Milestone 7 completes the independently resumable, hash-linked end-to-end recipe. A fresh locked
environment reproduced all 15 stages in 222.7 seconds with zero retries and a 7.41 GiB peak.
Milestone 8 completes the secure MiniSWE environment. Its Docker oracle solved 5/5 original tasks,
all five trajectories replayed exactly, and the retained capped Qwen baseline scored 0.3 after two
invalid JSON actions. Security probes confirmed non-root execution with no network, GPU, Docker
socket, capabilities, privilege escalation, or writable container root.

| Checkpoint | Protected exact accuracy | Parse rate |
| --- | ---: | ---: |
| Base | 0.0% | 0.0% |
| SFT | 86.4% | 95.5% |
| DPO | 84.1% | 95.5% |
| GRPO | 88.6% | 95.5% |

The NVIDIA RTX 4070 Ti SUPER 16 GB Linux x86-64 profile is **validated** for this pinned reference
path. See the [M7 completion report](docs/reference/m7-completion-report.md) and compact
[reference evidence](docs/reference/evidence/m7-reference-pipeline-92564f3.json). This measured
claim does not generalize automatically to other 16 GB GPUs.

The separate agent-environment validation is documented in the
[M8 completion report](docs/reference/m8-completion-report.md) and compact
[reference evidence](docs/reference/evidence/m8-reference-agent-9e3daa1.json).

## Foundation quick start

Review the [course prerequisites](docs/getting-started/prerequisites.md), then use Python 3.11 or
3.12:

```bash
uv sync --extra dev --extra docs
uv run nanopt --help
uv run nanopt config resolve \
  --hardware rtx_4070_ti_super_16gb \
  --model qwen3_0_6b_base \
  --experiment base_eval \
  --output resolved_config.yaml
uv run nanopt doctor --json doctor.json
uv run python labs/06_exact_generation.py
```

`nanopt doctor` is read-only and never downloads a model. On a CPU-only or non-reference machine it
will still write a report, then exit with the documented diagnostic status rather than claiming the
environment is usable for the reference pipeline.

## Design constraints

- Core SFT, DPO, and GRPO mathematics will live in readable NanoPT code.
- Transformers may load models and PEFT may inject and serialize LoRA adapters.
- Exact sampled token IDs are preserved for policy-gradient training; decoded text is never
  re-tokenized as the training trajectory.
- The required v0.1 path excludes Hydra, Ray, DeepSpeed, vLLM, FlashAttention, distributed runtimes,
  unrestricted agent shell access, and production RLHF abstractions.
- Public repository content is English, and hardware support is evidence-backed.

The detailed milestone contract is in the
[`implementation roadmap`](docs/12_IMPLEMENTATION_ROADMAP.md) and
[`acceptance criteria`](docs/13_ACCEPTANCE_CRITERIA.md).
The [20-chapter course map](docs/course/index.md) links each concept to its implementation, local
lab, and reference evidence. Start there after the prerequisites; use the
[observed-run troubleshooting guide](docs/troubleshooting.md) when a gate fails.

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md). From the repository root, run the complete M1 gate with:

```bash
./scripts/run_m1_gate.sh
```

The script stops at the first failure. It runs the following checks in order:

```bash
uv sync --frozen --extra dev --extra docs
uv run ruff format --check .
uv run ruff check .
uv run mypy src/nanopt
uv run pytest --cov=nanopt --cov-report=term-missing
uv run python scripts/validate_schemas.py
uv run python scripts/validate_m9_curriculum.py
uv run python scripts/lint_formulas.py docs
uv run mkdocs build --strict
uv build
```

NanoPT is licensed under Apache-2.0.
