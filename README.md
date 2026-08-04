# NanoPT

**Post-train a small language model into an assistant, a reasoner, and a tool-using agent—without
hiding the important parts.**

[![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11–3.12-3776AB.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-4C1.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/course-read%20online-7E57C2.svg)](https://shenli.github.io/nanopt/)
[![GPU](https://img.shields.io/badge/reference%20GPU-RTX%204070%20Ti%20SUPER-76B900.svg)](docs/08_HARDWARE_AND_PERFORMANCE.md)

NanoPT is an executable course and white-box reference implementation for modern language-model
post-training. It connects hand-computable equations to tested PyTorch, real LoRA training,
inspectable run artifacts, and a resettable coding-agent environment.

```text
Qwen3-0.6B Base
├── completion-only SFT → controlled DPO → synchronous GRPO/RLVR
└── replayed tool trajectories → exact-token Agent SFT → Mini Agent RL
```

## Why NanoPT?

- **Readable mathematics.** Masks, log probabilities, losses, accumulation, clipping, and
  checkpoint boundaries live in ordinary, commented Python.
- **Evidence before claims.** Examples and trajectories are written before aggregates. Hardware
  support is tied to retained measurements from a pinned commit.
- **Agents are stateful systems.** Tools, budgets, resets, recovery turns, context policy, hidden
  verification, and sandbox boundaries are first-class—not collapsed into prompt text.
- **Small enough to study.** The official path uses Qwen3-0.6B Base and one validated 16 GB consumer
  GPU. CPU labs cover every central invariant without a model download.

## v0.2 at a glance

NanoPT v0.2.0 adds a complete Agent SFT vertical slice. The clean reference gate on an NVIDIA RTX
4070 Ti SUPER measured:

| Evidence | Result |
| --- | ---: |
| Replay-checked source trajectories | 10/10 |
| Exact-token examples | 24 train / 6 task-held-out validation |
| Recovery examples | 5 |
| Held-out action-token accuracy | 76.5% → **95.0%** |
| Held-out action NLL | 1.240 → **0.363** |
| Base → adapted action validity | 0% → **100%** |
| Demonstrated / held-out Docker task score | **1.0 / 1.0** |
| Full transcript / snapshot validity | **100% / 75%** |
| Peak reserved VRAM | **13.94 GiB** |

The suite contains five deliberately tiny educational tasks. A 1/1 held-out result validates this
release protocol; it is not a claim of general software-engineering ability. See the
[Agent SFT report](docs/reference/v0.2-agent-sft-report.md) and
[compact evidence](docs/reference/evidence/v0.2-agent-sft-37acbc8.json).

The earlier math path remains fully executable. Its retained v0.1 reference results were:

| Checkpoint | Protected exact accuracy | Parse rate |
| --- | ---: | ---: |
| Base | 0.0% | 0.0% |
| SFT | 86.4% | 95.5% |
| DPO | 84.1% | 95.5% |
| GRPO | **88.6%** | 95.5% |

## Start on CPU

Use Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/shenli/nanopt.git
cd nanopt
uv sync --frozen --extra dev --extra docs

uv run nanopt --help
uv run nanopt doctor --json doctor.json
uv run python labs/01_tokens_and_masks.py
uv run python labs/20_agent_sft_masks.py
uv run python labs/21_agent_rl_credit.py
```

`nanopt doctor` is read-only. On unsupported hardware it reports what differs instead of claiming
the reference path is usable.

## Run Agent SFT

The data build freezes messages, token IDs, attention masks, current-action masks, prompt lengths,
chat-template hashes, and source-trajectory lineage:

```bash
uv run nanopt agent build-sft-data \
  --output artifacts/data/mini_swe_agent_sft_v1 \
  --local-files-only

uv run nanopt train agent-sft \
  --dataset artifacts/data/mini_swe_agent_sft_v1 \
  --local-files-only \
  --device cuda
```

Then evaluate the adapter in the network-disabled Docker environment:

```bash
uv run nanopt agent run \
  --policy model \
  --experiment agent_sft_eval \
  --task-split all \
  --task-id clamp_reversed_bounds \
  --context-policy full_transcript \
  --adapter artifacts/runs/AGENT_SFT_RUN/adapter/agent_sft \
  --adapter-name agent_sft \
  --local-files-only \
  --device cuda
```

Read [Agent SFT: from replayable trajectories to action targets](docs/agents/agent-sft.md) before
changing the data contract.

## Run Mini Agent RL

Agent RL starts from the v0.2 Agent SFT adapter. Model generation and optimization use the host
GPU; allow-listed tools and hidden verification remain inside the network-disabled Docker sandbox:

```bash
uv run nanopt train agent-rl \
  --agent-sft-adapter artifacts/runs/AGENT_SFT_RUN/adapter/agent_sft \
  --tasks-root tasks/mini_swe_v1 \
  --local-files-only \
  --device cuda
```

The run keeps exact prompt/action IDs, behavior and reference log probabilities, snapshot identity,
policy versions, and post-terminal hidden outcome rewards. It also writes fresh/stale,
credit-assignment, and tool-budget studies. Read [Mini Agent RL](docs/agents/agent-rl.md) before
changing rollout or policy-age semantics.

## Learn in layers

The [22-chapter course map](docs/course/index.md) links each idea to source code, a CPU lab, and the
appropriate evidence tier. A useful route is:

1. [Prerequisites](docs/getting-started/prerequisites.md)
2. [Tokens, masks, and log probabilities](docs/foundations/tokens-masks-logprobs.md)
3. [Completion-only SFT](docs/sft/completion-only-training.md)
4. [DPO](docs/preferences/dpo-training.md)
5. [Synchronous GRPO](docs/grpo-rlvr/synchronous-grpo.md)
6. [From tool calls to trajectories](docs/agents/from-tool-call-to-trajectory.md)
7. [Exact-token Agent SFT](docs/agents/agent-sft.md)
8. [Mini Agent RL](docs/agents/agent-rl.md)

## Repository map

```text
src/nanopt/core/       objectives and tensor invariants
src/nanopt/sft/        readable completion-only trainer
src/nanopt/dpo/        preference cache and DPO training
src/nanopt/grpo/       exact-token rollout and synchronous GRPO
src/nanopt/agent/      MiniSWE environment, replay, Agent SFT, and Agent RL
labs/                  22 executable local lessons
tasks/                 deterministic arithmetic and MiniSWE tasks
configs/               strict, mirrored experiment profiles
specs/schemas/         public artifact and dataset contracts
docs/reference/        measured reports and compact retained evidence
```

## Project promises

- Decoded rollout text is never re-tokenized for policy-gradient, Agent SFT, or Agent RL training.
- Prompt, prior-action, current-action, and padding coordinates remain explicit.
- Hidden verifier source and output never enter model observations or training datasets.
- The reference agent has allow-listed tools, no arbitrary shell, no network, no GPU, no Linux
  capabilities, a read-only container root, and separate hidden verification.
- Transformers loads models; PEFT injects and saves LoRA. The training objectives stay in NanoPT.
- GitHub Actions are intentionally absent. Validation is local and reproducible.

## Validate a checkout

```bash
./scripts/run_m1_gate.sh
```

The gate checks formatting, lint, strict typing, 400+ tests with coverage, schemas, the curriculum,
the public-release audit, documentation formulas, a strict MkDocs build, and wheel/source builds.
The longer GPU and Docker gates are explicit scripts under [`scripts/`](scripts/).

NanoPT is an alpha educational reference, not a production training framework. Distributed
runtimes, accelerated rollout servers, QLoRA, and long-horizon Agent RL remain later work. Contributions are
welcome through [`CONTRIBUTING.md`](CONTRIBUTING.md).

Licensed under [Apache 2.0](LICENSE).
