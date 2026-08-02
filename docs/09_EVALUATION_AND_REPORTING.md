# Evaluation and Reporting

## 1. Evaluation philosophy

Training loss and training reward are debugging signals, not proof of capability. NanoPT evaluates every checkpoint with the same versioned harness and stores example-level outputs before computing aggregates.

The evaluation design must answer:

- What capability changed at this stage?
- Did the model merely learn formatting or response length?
- Did one task family improve while another regressed?
- Did reward increase because the verifier can be exploited?
- How much did the policy drift from its parent?
- What did the improvement cost in memory and time?

## 2. Checkpoints compared

The standard report includes:

- Base;
- SFT;
- DPO;
- GRPO.

It may also include ablations such as SFT without completion masking, DPO beta variants, GRPO group-size variants, or KL variants. Every row must name its exact run and checkpoint lineage.

## 3. Evaluation modes

### 3.1 Deterministic pass@1

Use greedy decoding or an explicitly deterministic generation policy. Store the exact generation configuration. This measures the most likely response under a stable decoding path.

### 3.2 Sampled pass@1 and pass@k

Use a fixed seed schedule and documented sampling settings. For $n$ generated samples with $c$ correct and $n \ge k$, the standard estimator is:

$$
\operatorname{pass@k}
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

If exactly $k$ samples are generated, also report the direct “at least one passed” rate. Do not mix estimators without labels.

### 3.3 Teacher-forced metrics

On fixed target sequences, log completion NLL and optional token accuracy. These are useful for SFT and drift analysis but do not replace generation evaluation.

### 3.4 Preference evaluation

Evaluate held-out preference pairs with policy margin and preference accuracy. Include rejection-type breakdowns to detect shortcuts.

## 4. Required dataset splits

Report each separately:

- IID test;
- compositional test;
- range-shift test;
- format-attack test;
- optional anchor/conversational-format set.

Never combine protected test data into hyperparameter selection or reward construction.

## 5. Core metrics

### Capability

- exact-answer accuracy;
- parse rate;
- pass@k;
- accuracy by task family and difficulty;
- public/hidden test rate for agent tasks.

### Behavior

- completion length mean, median, and quantiles;
- maximum-length termination fraction;
- EOS fraction;
- number of answer fields;
- format compliance;
- repetition or duplicate-response rate;
- optional language/character-set diagnostics.

### Optimization

- loss and reward curves;
- DPO margin and preference accuracy;
- GRPO group reward variation and degenerate-group fraction;
- ratio and clip fraction;
- entropy or named entropy proxy;
- old-policy and reference-policy drift;
- gradient norms.

### Systems

- peak allocated and reserved VRAM;
- generated/training token throughput;
- wall-clock time;
- rollout/training/verifier time distribution;
- checkpoint size.

## 6. Statistical treatment

For accuracy-like metrics, include confidence intervals using a documented method such as Wilson intervals or bootstrap intervals. For checkpoint comparisons, use paired bootstrap resampling over the same task IDs when possible.

The report must not treat a one-point change on a small test set as definitive. Show task count and interval beside every headline metric.

## 7. Regression analysis

The report includes paired examples:

- Base wrong → later checkpoint correct;
- Base correct → later checkpoint wrong;
- parser invalid → valid;
- concise → excessive verbosity;
- correct result with invalid format;
- suspected reward hacking;
- DPO rejection-type failures;
- GRPO all-equal reward groups.

Selection rules must be deterministic, for example top absolute reward change followed by task ID, not cherry-picked manually.

## 8. Reward-hacking report

The format-attack suite probes:

- multiple answer tags;
- conflicting final answers;
- answer in an ignored field;
- `NaN`, infinity, and extremely large values;
- arithmetic expressions where a literal is required;
- trailing text that attempts to override the parser;
- code or markup intended to confuse extraction;
- correct substring inside a wrong answer;
- repeated answers with one correct candidate.

The report separates parser acceptance, verifier correctness, and reward received. Any case with reward inconsistent with trusted verification is a release blocker.

## 9. Drift metrics

On a fixed probe set, compute:

- parent-to-child completion NLL change;
- sampled-token log-probability difference;
- optional token-level KL estimate;
- answer-distribution changes;
- output length and entropy changes.

A LoRA parameter norm may be logged as a systems diagnostic but must not be presented as a substitute for behavioral drift.

## 10. Report artifacts

`nanopt report build` produces:

- `report.md`: readable in GitHub and diffable;
- `report.html`: local interactive or richly linked report with no server dependency;
- `summary.json`: machine-readable headline metrics;
- `plots/*.png` or SVG;
- tables that link to `samples.jsonl` and `trajectories.jsonl` record IDs.

The HTML report must not embed secrets or absolute local paths.

## 11. Proposed report structure

1. Run identity and reproducibility status.
2. Hardware and software environment.
3. Pipeline/checkpoint lineage.
4. Dataset and split fingerprints.
5. Headline checkpoint comparison.
6. SFT training and evaluation.
7. DPO training and preference analysis.
8. GRPO rollout/reward/optimization analysis.
9. Generalization and regression tables.
10. Reward-hacking analysis.
11. Resource usage and wall time.
12. Selected examples and trajectories.
13. Known limitations and failed runs.

## 12. Reference-result policy

Only results generated from a clean tagged commit, immutable model revision, protected test split, and validated hardware profile may be placed in the README reference table. Every number links to a manifest/checksum bundle.

Failed runs should be retained in a separate experiments log when they teach a useful lesson. Do not silently remove unstable or negative results from the course narrative.
