# Coding Agent: Start Here

You are implementing an open-source repository named **nanopt**. The prose brand is **NanoPT**, and the full name is **Nano Post-Training**.

## Mission

Build the smallest repository that is still complete enough to let a technically experienced learner:

1. inspect token-level log probabilities and masks;
2. fine-tune a small base language model with SFT;
3. construct preference pairs and train with DPO;
4. run synchronous GRPO/RLVR with a deterministic verifier;
5. compare checkpoints with reproducible evaluation and regression reports;
6. understand, run, and inspect a small stateful coding-agent environment;
7. complete the official path on one RTX 4070 Ti SUPER with 16 GB VRAM.

The repository is an **executable course and white-box reference implementation**, not a production RLHF framework.

## Non-negotiable constraints

- Everything in the public repository is in English.
- The import package and CLI command are `nanopt`.
- The reference model for the first complete pipeline is `Qwen/Qwen3-0.6B-Base`.
- The initial hardware profile is `rtx_4070_ti_super_16gb`.
- Core SFT, DPO, and GRPO mathematics must be implemented in readable project code rather than delegated entirely to a trainer library.
- Hugging Face Transformers may provide the model/tokenizer and PEFT may provide LoRA injection and adapter serialization.
- TRL may be used only in optional parity examples after the white-box implementation is correct.
- Do not introduce Hydra, Ray, DeepSpeed, vLLM, FlashAttention, or a distributed runtime into the required v0.1 path.
- Do not claim support for hardware that has not completed the validation protocol.
- Do not claim that a run fits in 16 GB until calibration and a full reference run prove it.
- Do not hide algorithmic control flow behind callbacks, factories, or a generic trainer abstraction.
- Do not use notebooks as the source of truth. Notebooks may be generated later from tested Python modules.
- Do not allow arbitrary host shell access in the agent environment.
- Do not expose hidden tests or verifier implementation details to the model.
- Do not decode generated text and then re-tokenize it for policy-gradient training. Preserve exact sampled token IDs.

## Implementation style

Follow these principles:

1. **Readable vertical slices.** A learner should be able to read each training entry point top to bottom.
2. **Small explicit functions.** Core math should be expressed as tensor functions with documented shapes.
3. **Tests before scale.** Every loss and masking rule needs a hand-computable unit test.
4. **Artifacts over anecdotes.** Every run records its resolved config, environment, metrics, samples, and git revision.
5. **Measured hardware support.** Hardware profiles are evidence-backed recipes, not marketing labels.
6. **Exact data lineage.** Generated datasets and preference pairs record generator version, seed, split, and task family.
7. **Secure-by-default environments.** Agent tools are allow-listed and sandboxes have no network by default.
8. **Course and code stay synchronized.** Documentation references stable functions and tests, not pasted duplicate implementations.

## Required workflow

Implement the repository milestone by milestone. Do not begin GRPO before the token-log-probability, mask, SFT, evaluation, and DPO foundations pass their acceptance tests.

For each milestone:

1. read the relevant specification;
2. write or update an ADR when the plan leaves a genuine choice open;
3. implement the smallest complete vertical slice;
4. add unit and integration tests;
5. run CPU smoke tests;
6. run the marked GPU smoke test when a compatible GPU is available;
7. update English documentation;
8. save evidence under `artifacts/reference/` only after the full validation protocol passes.

## First tasks

The first pull request should contain only the repository foundation:

- `pyproject.toml` and `uv.lock`;
- `src/nanopt/` package with version and CLI skeleton;
- `nanopt doctor`;
- typed config models and deterministic config resolution;
- the initial hardware/model/experiment config files;
- CPU-only unit-test infrastructure;
- MkDocs with working MathJax rendering;
- CI for lint, type checking, tests, and strict docs build;
- no training implementation yet.

The second pull request should implement tokenization, completion masks, exact token log-probabilities, model loading, baseline generation, and a synthetic arithmetic task generator.

## Publication decision status

- The repository identity is `shenli/nanopt`, matching the configured origin.
- The distribution name is `nanopt`; PyPI returned no project record on 2026-08-02, but availability must be checked again immediately before publication. If unavailable, keep the import and CLI as `nanopt` and use a distribution name such as `nanopt-llm`.
- Apache-2.0 is selected, matching the committed license and ADR-0000.
- The location for generated model adapters and datasets still requires owner confirmation before those artifacts are published.

## Definition of success for v0.1

A clean checkout on the reference machine can execute one documented command sequence that produces:

```text
base evaluation
→ SFT adapter and evaluation
→ DPO adapter and evaluation
→ GRPO adapter and evaluation
→ comparison report
```

The same checkout can run a small deterministic coding-agent task suite with resettable workspaces, allow-listed tools, public tests, hidden tests, trajectory logs, and no model training in the agent environment yet.

See `docs/13_ACCEPTANCE_CRITERIA.md` for the complete release gate.
