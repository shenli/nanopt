# Risks and Open Questions

## 1. Naming and publication

### Risk

`nanopt` or NanoPT may already be used as a package, repository, company/product name, or technical term in another field.

### Mitigation

Verify GitHub, PyPI, domain, and relevant trademark availability before publication. If the PyPI distribution is occupied, keep the import/CLI as `nanopt` and choose a distinct distribution name.

### Recorded decision

ADR-0000 selects `shenli/nanopt`, the `nanopt` distribution/import/CLI name, and Apache-2.0.
PyPI availability must still be rechecked immediately before the first upload because a lookup does
not reserve the name.

## 2. Upstream API churn

### Risk

Transformers, PEFT, TRL, PyTorch SDPA, and Qwen integrations change. A tutorial can become stale quickly.

### Mitigation

- lock dependencies;
- pin model revisions in validated releases;
- minimize dependence on trainer internals;
- have integration tests for renderer, adapter lifecycle, and generation;
- update with explicit migration notes.

## 3. Chat-template boundary correctness

### Risk

Using a tokenizer chat template can make prompt/completion boundary masking subtle. Template changes can invalidate caches and metrics.

### Mitigation

- renderer version and template hash in fingerprints;
- separate prompt/completion rendering tests;
- inspect exact token IDs in the first lab;
- never infer boundaries from decoded text.

## 4. Hardware fit and desktop VRAM

### Risk

The proposed configurations may fit in nominal 16 GB but fail due to display use, allocator fragmentation, driver differences, or longer examples.

### Mitigation

- conservative starting configs;
- phase-specific calibration;
- hard/soft memory budgets;
- no validated claim before full run;
- explicit OOM guidance.

## 5. Training signal too easy or too hard

### Risk

If every GRPO group is all correct or all wrong, group-relative advantages vanish. If tasks are trivial, the pipeline cannot demonstrate learning; if too hard, RL receives no signal.

### Mitigation

- difficulty-stratified prompt pool;
- log group reward variance and all-equal fraction;
- pilot runs before freezing reference distribution;
- preserve fixed anchor tasks;
- add curriculum only after baseline instrumentation works.

## 6. DPO shortcuts

### Risk

Chosen/rejected pairs may differ in length, formatting, or template artifacts so the model learns superficial preference signals.

### Mitigation

- controlled rejection categories;
- pair audits;
- matched-length variants;
- held-out category breakdowns;
- baseline heuristic classifiers;
- no claim based on training loss alone.

## 7. Reward hacking

### Risk

The model may exploit answer parsing, produce multiple candidates, or trigger verifier errors that accidentally yield reward.

### Mitigation

- strict parser separate from verifier;
- adversarial format split;
- fail-closed verifier status;
- raw response review;
- no reward on exceptions;
- release-blocking parser/reward consistency tests.

## 8. GRPO objective ambiguity

### Risk

Different projects use different advantage scaling, KL terms, clipping, and length normalization under the same algorithm name.

### Mitigation

Expose each choice in config, formula, metrics, and report. Add parity only after exact objective alignment.

## 9. Adapter reference/policy contamination

### Risk

Copying or switching PEFT adapters incorrectly may mutate the reference policy or use the wrong adapter for rollout/scoring.

### Mitigation

- immutable adapter hashes;
- explicit adapter context manager;
- reference `requires_grad=False` assertion;
- logits equality tests before training;
- active-adapter name in debug metrics;
- cache fingerprints.

## 10. Sampler and log-probability mismatch

### Risk

Temperature, top-p, token suppression, repetition penalties, or generation APIs may make stored log probabilities differ from the true behavior distribution.

### Mitigation

Use a simple custom reference sampler with temperature one and no truncation. Treat alternative sampling policies as separate backends with dedicated tests.

## 11. Evaluation contamination

### Risk

Protected tests may influence task generation, preference construction, reward shaping, or tuning.

### Mitigation

- immutable split manifests;
- one-way data flow;
- final test evaluation after recipe freeze;
- recorded access policy;
- no protected examples in docs.

## 12. Agent sandbox security

### Risk

Containerized model-generated code may attack the host, exfiltrate data, or consume resources.

### Mitigation

- structured tools only;
- no network/GPU;
- non-root and dropped capabilities;
- no Docker socket;
- path validation;
- resource limits and timeouts;
- separate verifier;
- run on a host with no sensitive mounts/secrets;
- clearly document that containers are not a perfect hostile-code boundary.

## 13. Task-suite validity

### Risk

MiniSWE tasks may be unsolvable, flaky, leak hidden expectations, or reward trivial hard-coding.

### Mitigation

- original small repositories;
- scripted oracle for every task;
- deterministic tests;
- hidden variants;
- task review checklist;
- repeat runs from clean snapshots.

## 14. Scope expansion

### Risk

The project may attempt SFT, DPO, reward modeling, PPO, GRPO, Agent RL, distributed infrastructure, and many models at once, becoming another incomplete framework.

### Mitigation

Hold v0.1 to the acceptance criteria. PPO remains a tiny lab; Agent RL and distributed backends remain future releases.

## 15. Performance versus readability

### Risk

A simple sampler may dominate runtime, creating pressure to add vLLM before correctness is stable.

### Mitigation

Profile first, preserve the reference backend forever, and add accelerated implementations behind identical trajectory schemas later.

## 16. Empirical release targets

### Open question

What exact quality thresholds should Base, SFT, DPO, and GRPO meet?

### Plan

Run pilot experiments, then freeze a `reference_targets.yaml` before final tuning. Targets should include parse rate, held-out accuracy/reward, preference metrics, reward-hacking integrity, and allowed anchor regressions.

## 17. Dataset and artifact publication

### Open questions

- publish generated data in the main repo, Git LFS, releases, or a separate dataset repository?
- publish adapters for every stage?
- how large should committed reference artifacts be?

### Proposed answer

Keep code/configs/small smoke data in git; publish large datasets, adapters, and full artifacts as versioned external releases with checksums and cards. Commit compact summary evidence and links.

## 18. Baseline operating system

### Open question

Should WSL2 be officially supported?

### Proposed answer

Begin with native Linux x86_64. Accept WSL2 reports as community-tested until Docker, filesystem, CUDA, and timing behavior pass the same evidence process.
