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
