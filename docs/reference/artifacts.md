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
