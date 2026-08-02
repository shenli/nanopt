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

A separate MiniSWE subsystem will teach resettable, allow-listed, stateful coding-agent
environments. v0.1 evaluates that environment; it does not train an agent policy in it.

## Current status

NanoPT is pre-alpha. Milestone 1 provides the package, CLI, typed configuration resolution,
environment diagnosis, run-artifact lifecycle, documentation skeleton, and local CPU validation.
Training and model-loading implementations are intentionally not present yet.

The only proposed reference profile is one NVIDIA RTX 4070 Ti SUPER with 16 GB VRAM on Linux
x86-64. This profile is **not validated**. No memory, runtime, performance, or hardware-support
claim should be inferred until a complete evidence bundle passes the release protocol.

## Foundation quick start

Use Python 3.11 or 3.12:

```bash
uv sync --extra dev --extra docs
uv run nanopt --help
uv run nanopt config resolve \
  --hardware rtx_4070_ti_super_16gb \
  --model qwen3_0_6b_base \
  --experiment base_eval \
  --output resolved_config.yaml
uv run nanopt doctor --json doctor.json
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

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The standard CPU checks are:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/nanopt
uv run pytest --cov=nanopt --cov-report=term-missing
uv run python scripts/validate_schemas.py
uv run python scripts/lint_formulas.py docs
uv run mkdocs build --strict
uv build
```

NanoPT is licensed under Apache-2.0.
