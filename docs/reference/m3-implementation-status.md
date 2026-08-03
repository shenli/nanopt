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

The final local gate collected 324 tests: 323 passed and one opt-in network tokenizer test was
skipped. Total branch-aware source coverage was 89%. Ruff formatting/linting, strict typing across
45 source files, eight JSON schemas, ten YAML profiles, 44 formula-linted Markdown files, the strict
documentation build, and both package distributions passed. The opt-in pinned real-tokenizer test
also passed separately, including EOS removal from parser-facing decoded text.

## Reference smoke still required

On the RTX 4070 Ti SUPER host, check out the current `main` commit and run:

```bash
bash scripts/run_m3_reference_smoke.sh
```

The first environment probe caught a real compatibility boundary before model loading: the broad
PyTorch range had resolved a CUDA 13 wheel that could not initialize with the host's NVIDIA 560
driver. [ADR-0004](../adr/0004-reference-pytorch-wheel.md) records the resulting PyTorch 2.7.1 /
CUDA 12.6 platform pin. This is a useful example of why dependency locks and `doctor` checks are
part of the experiment, not merely installation housekeeping.

The corrected probe also showed why the hardware matcher allows a small difference between a
board's marketed capacity and the bytes PyTorch exposes after driver/runtime reservations. Recipe
memory limits remain explicit and stricter than this product-identity check.

The script refuses a dirty checkout, creates a unique ignored directory under `artifacts/tmp/`, and
runs the locked environment sync, hardware diagnosis, deterministic data generation, real model
load, two-example calibration, and full deterministic baseline. It then runs
`scripts/validate_m3_reference_smoke.py`, which checks:

- Linux/x86-64, one matching RTX 4070 Ti SUPER, CUDA, and recorded driver/runtime;
- task counts and canonical hashes against the split manifest;
- the dataset fingerprint in both evaluation manifests;
- clean Git identity and pinned model/tokenizer revisions;
- representative versus calibration labels and expected task counts;
- JSON schemas, example uniqueness, artifact checksums, and report leakage rules.

Review the generated `m3_smoke_evidence.json`, resolved configs, manifests, environments,
`samples.jsonl` files, summaries, and reports. If the validator passes without source edits, commit a
small reviewed evidence summary and change M3 status to complete. This smoke run still does not
validate the full training pipeline or the final 16 GB support claim.
