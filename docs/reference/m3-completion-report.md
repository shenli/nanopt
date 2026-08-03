# M3 completion report

Milestone 3 is complete. Its CPU/fixture gate and one real base-model evaluation passed on the
proposed reference machine. This closes the milestone's generation-and-evaluation exit criterion;
it does **not** validate the later training pipeline or make the hardware profile supported.

## Delivered

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

## Local acceptance evidence

The final local gate collected 325 tests: 324 passed and one opt-in network tokenizer test was
skipped. Total branch-aware source coverage was 89%. Ruff formatting/linting, strict typing across
45 source files, eight JSON schemas, ten YAML profiles, 45 formula-linted Markdown files, the strict
documentation build, and both package distributions passed. The pinned real-tokenizer test had
also passed separately, including EOS removal from parser-facing decoded text.

Focused tests cover token-by-token log probabilities, first-EOS masks, seed independence, raw
policy versus sampling-behavior probabilities, hand-computed statistical fixtures, artifact write
ordering, report safety, deterministic data generation, and reference-evidence tamper rejection.

## Reference smoke evidence

The reference command completed at 2026-08-03 UTC from clean commit
`8de4507234b51c3f105e1418f99c3d51ac59fee9`:

```bash
bash scripts/run_m3_reference_smoke.sh
```

| Field | Observed value |
| --- | --- |
| Host | Linux x86-64 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, compute capability 8.9 |
| Driver / CUDA / PyTorch | 560.35.03 / 12.6 / 2.7.1+cu126 |
| Model | `Qwen/Qwen3-0.6B-Base` at revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Parameters | 596,049,920 |
| Dataset | 128 records; fingerprint `83225e59551f147cdd90e854570dec56dd386a96b7c9175c514bd0033d7315ff` |
| Calibration | 2 examples, explicitly non-representative |
| Baseline evaluation | 44 held-out examples, deterministic CUDA generation |

The offline validator checked the machine report, clean Git identity, pinned model and tokenizer
revisions, dataset hashes and split counts, run/sample schemas, representative labels, artifact
checksums, task uniqueness, and report leakage rules. Its compact, reviewable output is preserved
in [the M3 reference evidence summary](evidence/m3-reference-smoke-8de4507.json).

## Baseline result and interpretation

The base checkpoint produced no parseable exact answers: accuracy, parse rate, EOS fraction, and
pass@1 were all 0% over 44 examples. Every completion reached the 128-token limit. The 95% Wilson
upper bound for accuracy and parse rate was 8.0%.

That result is not a gate failure. M3 asks whether the real checkpoint can be loaded, evaluated,
and audited—not whether an untrained base checkpoint already follows the course's answer protocol.
The poor baseline is useful: it makes format learning an observable SFT objective. It also means M4
should inspect example generations and the rendering contract before attributing any later gain
solely to optimization.

## Problems caught by the gate

The first environment probe resolved a CUDA 13 PyTorch build that the host's NVIDIA 560 driver
could not initialize. [ADR-0004](../adr/0004-reference-pytorch-wheel.md) records the resulting
PyTorch 2.7.1 / CUDA 12.6 platform pin. The next probe showed that PyTorch exposes slightly less
than the board's marketed capacity after driver/runtime reservations; the product matcher now
allows that small difference while recipe memory budgets remain explicit and stricter.

These failures occurred before evidence collection and were fixed, locally gated, committed, and
re-run from a clean checkout. They demonstrate why environment diagnosis is part of the experiment.

## Remaining boundary

No SFT, DPO, GRPO, peak-memory, throughput, resume, or end-to-end pipeline claim follows from this
smoke. The `rtx_4070_ti_super_16gb` profile remains `proposed_unvalidated` until the full hardware
protocol passes. The next milestone is the readable SFT vertical slice in M4.
