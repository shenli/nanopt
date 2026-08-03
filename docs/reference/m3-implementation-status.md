# M3 implementation status

Milestone 3 is implemented and validated on the local CPU/fixture tier. It is **not complete**
because its exit criterion requires one base-model evaluation on the proposed reference machine.
No GPU, memory-fit, throughput, runtime, accuracy, or hardware-support claim is made here.

## Delivered locally

| Roadmap item | Inspectable implementation |
| --- | --- |
| Exact sampler | `nanopt.rollout.sampler.sample_autoregressive` |
| EOS/stopping masks | `nanopt.rollout.stopping` |
| Deterministic and sampled evaluation | `nanopt.eval.runner` and `nanopt eval run` |
| Stable seeds | SHA-256 task/sample seed schedule |
| pass@k and intervals | `nanopt.eval.metrics` |
| Example JSONL | Strict `EvaluationResult`, appended before aggregation |
| Checkpoint-neutral interface | `EvaluationBackend` protocol plus local model adapter |
| Reports | Offline Markdown, HTML, and JSON summary builder |
| Baseline profile | `configs/experiments/base_eval.yaml` |
| Calibration | `nanopt calibrate --mode load` and `--mode eval` |
| Learner path | Exact-generation chapter and CPU lab 06 |

## Acceptance evidence

Focused tests cover:

- token-by-token log probabilities versus teacher-forced scoring;
- greedy EOS termination and first-EOS masks;
- sampled-seed reproducibility independent of global RNG state;
- raw-policy versus temperature/top-p behavior probabilities;
- hand-computed pass@k and Wilson edge cases;
- example-before-summary artifact ordering;
- deterministic report construction from fixture JSONL;
- rejection of absolute-path and secret-like report identities;
- deterministic dataset generation through the public CLI.

The final local gate collected 320 tests: 319 passed and one opt-in network tokenizer test was
skipped. Total branch-aware source coverage was 89%. Ruff formatting/linting, strict typing across
45 source files, eight JSON schemas, ten YAML profiles, 44 formula-linted Markdown files, the strict
documentation build, and both package distributions passed. The opt-in pinned real-tokenizer test
also passed separately, including EOS removal from parser-facing decoded text.

## Reference smoke still required

On the RTX 4070 Ti SUPER host:

```bash
uv sync --frozen --extra dev --extra docs
uv run nanopt doctor --json artifacts/m3-doctor.json
uv run nanopt data generate
uv run nanopt calibrate --mode load --device cuda
uv run nanopt calibrate \
  --mode eval \
  --tasks artifacts/data/arithmetic_v1/tasks.jsonl \
  --device cuda
uv run nanopt eval run \
  --tasks artifacts/data/arithmetic_v1/tasks.jsonl \
  --mode deterministic \
  --checkpoint-id base \
  --device cuda
```

Review the resolved config, manifest, environment, `samples.jsonl`, summary, and both reports. If the
run is fully inspectable and no source edits were needed, record the smoke evidence and change M3
status to complete. This smoke run still does not validate the full training pipeline or 16 GB
hardware claim.
