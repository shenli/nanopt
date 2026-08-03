# M2 completion report

Date: 2026-08-02

## Scope

M2 establishes the mathematical, model-integration, and deterministic-data dependencies required
before generation and baseline evaluation. It does not implement a sampler, an evaluation runner,
training loops, reference-hardware calibration, or a validated quality result.

## Delivered

### Core mathematics

- Exact causal token shifting and FP32 selected-token log probabilities.
- Full-token completion/action masks with prompt, padding, terminal-token, and post-terminal rules.
- FP32 masked sum/mean reductions with explicit zero-active-token failures.
- Full-vocabulary entropy and exact categorical KL.
- Direct sampled KL and nonnegative sampled k3 KL with exponent diagnostics.
- DPO policy/reference margins, implicit reward margin, and stable logistic loss.
- Group-centered and population-standardized group-relative advantages with degenerate-group flags.
- Current/old probability ratios and PPO-style clipping for both advantage signs.
- Explicit token-mean and sequence-mean policy-loss normalization.

### Model integration

- Qwen3 0.6B Base loader with explicit safe loading options and immutable model/tokenizer revisions.
- Canonical base profile pinned to Hugging Face commit
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
- Chat-template renderer that requests tensor output and proves the prompt is an exact prefix of the
  supervised sequence before constructing its action mask.
- Chat-template SHA-256 capture and explicit thinking-mode behavior.
- LoRA target validation, named adapter creation, parameter counts, clone, freeze, temporary
  selection, safetensors save, and local load.

### Synthetic data and verification

- Deterministic structured arithmetic AST generator for addition/subtraction, multiplication, exact
  division, and mixed fully parenthesized expressions.
- Exact trusted evaluation with `Fraction`; generated or model text is never passed to Python
  `eval`.
- Versioned strict task, target, provenance, verifier, and split-manifest records.
- Canonical task hashes before prompt rendering and complete dataset fingerprints.
- Deterministic seven-way split builder that rejects canonical overlap and assigns every task once.
- Strict final-answer parser with integer/rational canonicalization and an adversarial attack suite.
- Exact verifier that independently re-evaluates trusted AST state before comparing a candidate.
- Checked-in generator configuration and dataset card under `tasks/arithmetic/`.

### Course material

- CPU labs for prerequisites, tokens/masks, log probabilities, DPO, group advantages/clipping, and
  the complete synthetic-data path.
- Learner chapters for token coordinates, entropy/KL, model rendering/LoRA, DPO, group clipping, and
  deterministic arithmetic data.
- Integration tests execute every documented CPU lab from the repository root.

## Acceptance mapping

| M2 requirement | Evidence |
|---|---|
| Causal log probabilities, masks, reductions | `src/nanopt/core/{logprobs,masks,reductions}.py`; hand fixtures under `tests/unit/core/` |
| Entropy and KL | `src/nanopt/core/{entropy,kl}.py`; exact and sampled fixtures |
| DPO and clipping | `src/nanopt/core/{dpo,clipping}.py`; sign, beta, normalization, gradient tests |
| Group-relative advantages | `src/nanopt/core/advantages.py`; population-std and equal-group tests |
| Qwen loader and renderer | `src/nanopt/models/{loading,renderer}.py`; mocked loader and real pinned-tokenizer boundary tests |
| LoRA lifecycle | `src/nanopt/models/adapters.py`; tiny local Qwen3 adapter clone/save/load logit parity |
| Arithmetic generator | `src/nanopt/data/arithmetic.py`; multi-family property tests over deterministic seeds |
| Schemas and fingerprints | `src/nanopt/data/{schemas,fingerprints}.py`; JSON Schema and repeatability tests |
| Leakage-safe splits | `src/nanopt/data/splits.py`; canonical duplicate and disjoint-manifest tests |
| Strict parser/verifier | `src/nanopt/eval/{parser,verifier}.py`; malformed/multiple/trailing/non-finite attack tests |
| Required CPU labs | `labs/01` through `labs/04`; commands executed by `tests/integration/test_labs.py` |

## Verification

The required local gate is:

```text
./scripts/run_m1_gate.sh
```

The opt-in real-tokenizer boundary check is:

```text
NANOPT_RUN_NETWORK_TESTS=1 uv run pytest tests/network/test_qwen_tokenizer.py -q
```

Results:

- 279 local tests passed and the opt-in network test was skipped by default;
- one separately executed real pinned-tokenizer network test passed;
- total statement/branch coverage report: 92%;
- 36 package source files passed strict mypy;
- eight JSON schemas validated and ten public configuration profiles parsed;
- six documented CPU labs executed successfully through integration tests;
- 42 Markdown files passed formula lint and the strict documentation build;
- source distribution and universal wheel built successfully;
- the deterministic 20-task lab fingerprint was
  `7a11ded13b590273b10812f96889bf0a61552690f431d0b8b3ad12e7e52177d0`.

## CPU, network, and GPU status

All mathematical, renderer, adapter, generator, split, parser, verifier, lab, schema, and package
tests run on CPU. The adapter round trip uses a randomly initialized tiny Qwen3 model created from
local configuration and downloads no weights.

The official pinned Qwen tokenizer was downloaded and its exact supervised-prefix boundary test
passed. The 0.6B model weights were not downloaded or executed in this milestone. No CUDA/GPU,
memory-fit, runtime-performance, training, or reference-hardware claim is made.

## Next milestone

M3 may now build exact autoregressive sampling, stopping/EOS masks, baseline evaluation, pass@k and
confidence intervals, example-level artifacts, reports, and load/evaluation calibration on these
tested contracts.
