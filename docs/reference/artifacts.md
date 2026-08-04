# Run artifact contract

The M1 `RunContext` creates an inspectable directory before expensive work starts:

```text
artifacts/runs/<run-id>/
├── resolved_config.yaml
├── config_provenance.yaml
├── run_manifest.json
├── environment.json
├── metrics.jsonl
├── events.jsonl
├── samples.jsonl
├── summary.json
├── report.md
├── report.html
├── checkpoints/
├── cache/
└── plots/
```

JSON and YAML documents are atomically replaced. JSONL events are appended as complete records and
identify malformed lines if an external interruption leaves a partial write. Manifests capture the
Git revision and dirty state without user names, host names, local checkout paths, or credentialed
remote URLs.

Evaluation writes each typed sample before it contributes to `summary.json`. The report builder can
therefore reproduce aggregates after the model process exits. `report.md` and `report.html` contain
safe identifiers, aggregate metrics, and relative artifact links only; response bodies remain in
`samples.jsonl`.

Evaluation manifests record three data-lineage values: the complete generated-dataset fingerprint,
the task JSONL checksum, and the split-manifest checksum. Evaluation refuses a task file whose
counts or canonical hashes differ from its sibling `dataset_manifest.json`.

An end-to-end run adds a parent `pipeline_manifest.json`. It stores every logical stage and retained
attempt, the child run-manifest hash, input/output checkpoint hashes, wall time, phase memory peak,
and failure/retry disclosure. Resume re-hashes completed children and outputs before skipping them.
The final `comparison.json`, `report.md`, and `report.html` are rebuildable from child artifacts
without loading a model.

An agent-evaluation run adds `trajectories.jsonl`, one typed trajectory per task under
`agent_trajectories/`, inspectable diffs under `final_patches/`, and `replay.json` for scripted-oracle
semantic replay. `run_manifest.json` records the sandbox/backend policy, exact container image,
isolation flags, task-suite fingerprint, and the explicit fact that the environment did not train
the model. Hidden verifier output and source are absent by contract.

An Agent SFT dataset directory contains `manifest.json`, `examples.jsonl`, and hashed records under
`source_trajectories/`. Each example stores messages, completion, target action, token IDs,
attention mask, action mask, prompt length, context policy, chat-template hash, and its exact source
trajectory hash. An `agent_sft` run uses the normal metrics/checkpoint/adapter contract and adds the
dataset, example, and source-trajectory fingerprints to `run_manifest.json`.

An `agent_rl` run adds `rollout_groups.jsonl`, `staleness_study.json`, `credit_study.json`, and
`tool_budget_study.json`. Each action record retains the online prompt IDs, sampled IDs, aligned
mask, behavior log probabilities, reference log probabilities, parse status, tool, and episode
advantage. Group records prove task/snapshot/policy-version identity. Hidden verifier source and
output are absent; only bounded outcome counts and reward are retained after termination.
The run manifest and summary distinguish the terminal training version from the selected
post-update policy boundary, and retain validation reward for every version so a later regression
cannot disappear when an earlier adapter is published.

A `systems_lab` run is CPU-only and adds `actions.jsonl`, `partial_checkpoints.jsonl`,
`weight_sync_events.jsonl`, and `admission_decisions.jsonl`. Its synthetic action IDs expose the
control-plane record shape; they are never labeled as model output. `summary.json` and `report.md`
compare cache reuse, recomputation, staleness, and mixed-policy behavior. Both the config and
summary declare that simulated experience was not used for a model update and that no throughput
was measured.
