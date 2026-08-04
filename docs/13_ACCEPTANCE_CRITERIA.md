# Acceptance Criteria

This document is the release contract for NanoPT v0.1. A feature is not complete because its code exists; it must pass the relevant behavioral, reproducibility, documentation, and hardware gates.

## 1. Repository-wide gates

- [x] Public repository content is English, excluding proper names, quoted source titles, and intentionally multilingual test fixtures.
- [x] Package installs from `uv.lock` in a clean Linux environment.
- [x] `nanopt --help`, `nanopt doctor`, and `nanopt config resolve` work without downloading a model.
- [x] Ruff, type checking, CPU tests, package build, schema validation, and strict docs build pass.
- [x] No required golden-path command depends on notebooks, hosted telemetry, paid APIs, or private credentials.
- [x] README claims distinguish proposed, smoke-tested, and validated behavior.
- [x] Every reference result links to an evidence manifest and immutable commit/model revision.
- [x] No secret, absolute personal path, or hidden-test content appears in committed artifacts.

## 2. Mathematical correctness gates

- [x] Causal token shifting matches hand-computable fixtures.
- [x] Prompt, padding, and post-EOS tokens are excluded by masks.
- [x] Core reductions fail clearly on zero active tokens.
- [x] Log-softmax and loss reductions use the documented precision.
- [x] DPO sign, margin, beta, masking, and cache-parity tests pass.
- [x] Group-relative advantages match hand calculations and sum to approximately zero.
- [x] All-equal reward groups yield zero advantages.
- [x] PPO/GRPO clipping tests cover positive and negative advantages.
- [x] Sampler log probabilities match a direct forward pass in the reference sampling mode.
- [x] Loss-normalization variants have separate tests and names.
- [x] Optional KL estimator matches its formula and non-negativity expectations within numerical tolerance.

## 3. Data gates

- [x] Generated data validates against versioned schemas.
- [x] Identical generator config/seed produces identical content fingerprint.
- [x] No canonical task hash overlaps protected splits.
- [x] Every trusted answer is independently re-evaluated from the AST.
- [x] Every chosen preference passes the strict verifier.
- [x] Every rejected preference fails the intended criterion.
- [x] Preference audit reports length and rejection-type distributions.
- [x] Parser attack suite has zero cases where an incorrect trusted answer receives correctness reward.
- [x] Dataset cards explain scope, generation, limitations, and license.

## 4. Baseline/evaluation gates

- [x] Base model evaluation saves example-level outputs and aggregate metrics.
- [x] Deterministic and sampled generation modes are separately configured.
- [x] pass@k implementation passes fixtures.
- [x] Confidence intervals and task counts accompany headline accuracy metrics.
- [x] Reports separate parser failure from wrong answer.
- [x] Same evaluator version is used across checkpoint comparisons.
- [x] Regression selection is deterministic and reproducible.

## 5. SFT gates

- [x] Only intended LoRA parameters are trainable.
- [x] Prompt-only tokens contribute zero SFT loss.
- [x] A tiny repeated-batch test lowers completion loss.
- [x] Checkpoint save/load preserves logits within tolerance.
- [x] Resume from a clean optimizer boundary passes its fixture.
- [x] Reference SFT calibration stays below the hard VRAM budget.
- [x] Full SFT run completes on the target hardware without source edits.
- [x] Protected evaluation demonstrates that the model learned the answer protocol and did not merely reduce teacher-forced loss.

## 6. DPO gates

- [x] Frozen SFT reference cache is complete and fingerprinted.
- [x] Cached and live reference values agree on a sample.
- [x] DPO starts from an exact SFT policy copy.
- [x] A tiny controlled update increases the chosen margin.
- [x] Full DPO run completes on the target hardware.
- [x] Held-out preference loss/margin improves relative to the SFT policy.
- [x] Rejection-type breakdown shows no single unacknowledged shortcut dominates.
- [x] Protected task accuracy and parse rate regressions are reviewed and documented.

## 7. GRPO gates

- [x] Every rollout stores exact token IDs, action masks, old log probabilities, finish reason, and rewards.
- [x] Training consumes stored tokens without decode/re-tokenize.
- [x] Group-size invariant `G >= 2` is enforced.
- [x] Degenerate-group fraction is logged.
- [x] Non-finite rewards, advantages, ratios, losses, and gradients are detected.
- [x] Full calibration covers generation, verification, scoring, backward, and optimizer phases.
- [x] Full GRPO run completes on the reference hardware.
- [x] The report identifies the exact advantage, clipping, normalization, and KL variants.
- [x] Protected held-out expected reward or exact-answer accuracy improves over the DPO parent on at least one primary target split.
- [x] Any regression on an anchor/generalization split is quantified and judged against a threshold frozen before the final reference run.
- [x] Reward-hacking suite remains sound after training.

The project should define numeric release targets in a versioned `reference_targets.yaml` after pilot runs and before final reference tuning. Once frozen for a release candidate, those targets cannot be loosened without an ADR and a new candidate.

## 8. End-to-end pipeline gates

- [x] One documented command sequence runs Base → SFT → DPO → GRPO as independently resumable stages.
- [x] Pipeline manifest records every child run and checkpoint hash.
- [x] A fresh clone can rebuild the final comparison report from saved artifacts.
- [x] Model and tokenizer revisions are immutable in the evidence bundle.
- [x] Protected test data is not used for training, reward shaping, or final hyperparameter selection.
- [x] Total wall time and phase-specific VRAM peaks are measured, not estimated.
- [x] Failed/retried stages and deviations are disclosed.

## 9. Hardware-validation gates

- [x] `nanopt doctor` identifies the RTX 4070 Ti SUPER and records actual VRAM/driver/runtime.
- [x] BF16 support is checked at runtime.
- [x] Model load, evaluation, SFT, DPO, and GRPO calibrations pass.
- [x] Full pipeline stays below the profile hard memory budget.
- [x] No hidden manual source change is needed between stages.
- [x] Evidence bundle contains configs, manifests, metrics, reports, and checksums.
- [x] Profile status changes to `validated` in the same reviewed commit/release that contains evidence.

## 10. Agent-environment gates

- [x] Every task has an immutable initial snapshot and deterministic reset hash.
- [x] A scripted oracle solves every task and hidden verifier passes.
- [x] Model cannot supply arbitrary shell commands.
- [x] Path traversal, symlink escape, test modification, output overflow, timeout, and network attempts are blocked or safely contained.
- [x] Public tests and hidden verifier use separate workspaces.
- [x] Hidden test source never appears in model observations or public artifacts.
- [x] Trajectory records include task/environment/model versions, actions, results, budgets, and final score.
- [x] Docker reference backend runs non-root with no network/GPU and documented resource limits.
- [x] Environment report states that v0.1 evaluates agents but does not train them.

## v0.2 Agent SFT

- [x] Online inference and offline examples share one multi-turn context contract.
- [x] Stored examples retain exact token IDs and current-action masks in full coordinates.
- [x] Every example links to a hashed, replay-checked source trajectory.
- [x] Tool demonstrations cover inspection, editing, tests, finishing, and invalid-action recovery.
- [x] Agent SFT trains only LoRA parameters and saves optimizer-boundary checkpoints.
- [x] A task-level held-out split is excluded from training.
- [x] Teacher-forced action NLL and token accuracy improve on the held-out task.
- [x] Docker behavior reports action validity separately from hidden-verifier task score.
- [x] Full-transcript and observation-snapshot policies are compared explicitly.
- [x] The clean reference run remains under the RTX 4070 Ti SUPER hard VRAM limit.

## v0.3 Mini Agent RL

- [x] Every rollout group independently resets one immutable task snapshot.
- [x] Every action retains its exact online prompt IDs, sampled IDs, mask, and FP32 behavior log probabilities.
- [x] Hidden verifier source, output, and reward are absent from model observations.
- [x] The hidden terminal score is assigned only after episode termination.
- [x] Group-relative advantages and degenerate groups have hand-computable tests.
- [x] Training consumes stored IDs directly and rejects any nonzero policy lag.
- [x] The clipped objective, KL estimator, normalization, and update epoch count are explicit.
- [x] Fresh and stale retained groups are rescored under the final policy and excluded from updates.
- [x] All-action and terminal-action-only credit coverage are compared.
- [x] Parent and selected Agent RL policies are compared under at least two tool budgets on held-out tasks.
- [x] A clean Docker/GPU reference run stays under the hard VRAM limit and retains compact evidence.

## v0.4 Systems laboratory

- [x] Partial rollouts pause only between complete structured actions.
- [x] Every checkpoint pairs exact model execution state with external world state.
- [x] Snapshot, workspace, cursor, budget, and payload tampering stop resume.
- [x] Policy publication compares episode-boundary and action-boundary synchronization.
- [x] Prefix-cache identity includes both exact prompt IDs and behavior-policy hash.
- [x] Cache hits, misses, evictions, and synthetic recomputation costs are inspectable.
- [x] Admission records distinguish fresh, stale, and mixed-policy episodes.
- [x] A bounded-action counterfactual never silently becomes an implemented off-policy objective.
- [x] Simulated experience is explicitly excluded from model updates.
- [x] The CLI writes actions, checkpoints, sync events, admission decisions, summary, and report.
- [x] Documentation distinguishes CPU control-plane evidence from accelerated-runtime claims.

## 11. Documentation gates

- [x] Every chapter follows the standard chapter template where applicable.
- [x] All formulas render correctly with MathJax.
- [x] No malformed raw formula delimiters remain.
- [x] Every lab command has been run in its claimed environment tier.
- [x] Source links favor primary papers, official reports, and official repositories.
- [x] The course distinguishes source-supported facts, project design decisions, and unvalidated hypotheses.
- [x] Small-scale simplifications are explicitly mapped to industrial systems.
- [x] README provides a short path; detailed theory stays in docs.

These gates were closed by the clean M9 curriculum run recorded in the
[completion report](reference/m9-completion-report.md). Reference-tier commands retain their
accepted M3–M8 evidence rather than being mislabeled as CPU curriculum runs.

## 12. Release decision

A release candidate receives one of three outcomes:

- **Pass:** every required gate passes.
- **Pass with disclosed limitation:** only a non-core optional feature fails, and the claim is removed or narrowed.
- **Fail:** any mathematical, data leakage, verifier, security, reproducibility, or supported-hardware gate fails.

A failed core gate must not be converted into documentation wording that implies completion.
