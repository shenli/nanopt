# NanoPT

**Post-train a small language model into an assistant, a reasoner, and a tool-using agent—without
hiding the important parts.**

[![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11–3.12-3776AB.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-4C1.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/course-read%20online-7E57C2.svg)](https://shenli.github.io/nanopt/)
[![GPU](https://img.shields.io/badge/GPU-single%2016%20GB%20consumer%20card-76B900.svg)](docs/08_HARDWARE_AND_PERFORMANCE.md)

NanoPT is an executable course and white-box reference implementation for modern language-model
post-training. It connects hand-computable equations to tested PyTorch, real LoRA training,
inspectable run artifacts, and a resettable coding-agent environment.

```text
Qwen3-0.6B Base
├── completion-only SFT → controlled DPO → synchronous GRPO/RLVR
└── replayed tool trajectories → exact-token Agent SFT → Mini Agent RL → systems lab
```

## Why NanoPT?

- **Readable mathematics.** Masks, log probabilities, losses, accumulation, clipping, and
  checkpoint boundaries live in ordinary, commented Python.
- **Evidence before claims.** Examples and trajectories are written before aggregates. Hardware
  support is tied to retained measurements from a pinned commit.
- **Agents are stateful systems.** Tools, budgets, resets, recovery turns, context policy, hidden
  verification, and sandbox boundaries are first-class—not collapsed into prompt text.
- **Consumer hardware by design.** The official path uses Qwen3-0.6B Base and is validated on one
  consumer GPU with 16 GB VRAM; it does not require an H100 or B200. CPU labs cover every central
  invariant without a model download.

## Agent RL on one consumer GPU

NanoPT includes an exact-token Mini Agent RL vertical slice built on Agent SFT. A clean validated
run on one consumer GPU with 16 GB VRAM measured:

| Evidence | Result |
| --- | ---: |
| Rollout groups / episodes / action turns | **4 / 16 / 80** |
| Non-degenerate groups | **4/4** |
| Mean sampled outcome reward | **0.6719** |
| Sampled action validity | **91.25%** |
| Exact optimizer steps | **20** |
| Selected post-update validation reward | **1.0** |
| Peak reserved VRAM | **14.094 GiB** |

The suite contains five deliberately tiny educational tasks and one validation task. Policy
quality was not monotonic: the frozen selection rule published version 1, while the terminal
version scored 0.0. The full history stays visible as an example of why the last update is not
automatically the best checkpoint. See the [Mini Agent RL report](docs/reference/v0.3-agent-rl-report.md)
and [compact evidence](docs/reference/evidence/v0.3-agent-rl-85ca98b.json).

The earlier math path remains fully executable. Its retained reference results were:

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
uv run python labs/22_resumable_rollouts.py
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

## Run the v0.4 systems laboratory

Trace partial rollout state, policy publication, cache identity, and training admission without a
model download or cluster:

```bash
uv run python labs/22_resumable_rollouts.py
uv run nanopt systems simulate --run-id systems-tutorial
```

The simulation compares keeping old weights for a complete episode with synchronizing only
between complete tool actions. It writes inspectable model/world checkpoints and explicitly does
not use synthetic experience for training. Read [Reinforcement Learning from a Systems Perspective](docs/tutorials/rl-from-systems-perspective.md)
for the end-to-end walkthrough.

## Learn in layers

The [23-chapter course map](docs/course/index.md) links each idea to source code, a CPU lab, and the
appropriate evidence tier. A useful route is:

1. [Prerequisites](docs/getting-started/prerequisites.md)
2. [Tokens, masks, and log probabilities](docs/foundations/tokens-masks-logprobs.md)
3. [Completion-only SFT](docs/sft/completion-only-training.md)
4. [DPO](docs/preferences/dpo-training.md)
5. [Synchronous GRPO](docs/grpo-rlvr/synchronous-grpo.md)
6. [From tool calls to trajectories](docs/agents/from-tool-call-to-trajectory.md)
7. [Exact-token Agent SFT](docs/agents/agent-sft.md)
8. [Mini Agent RL](docs/agents/agent-rl.md)
9. [Reinforcement Learning from a Systems Perspective](docs/tutorials/rl-from-systems-perspective.md)

## Repository map

```text
src/nanopt/core/       objectives and tensor invariants
src/nanopt/sft/        readable completion-only trainer
src/nanopt/dpo/        preference cache and DPO training
src/nanopt/grpo/       exact-token rollout and synchronous GRPO
src/nanopt/agent/      MiniSWE environment, replay, Agent SFT, and Agent RL
labs/                  23 executable local lessons
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
