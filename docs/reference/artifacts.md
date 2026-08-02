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
├── checkpoints/
├── cache/
└── plots/
```

JSON and YAML documents are atomically replaced. JSONL events are appended as complete records and
identify malformed lines if an external interruption leaves a partial write. Manifests capture the
Git revision and dirty state without user names, host names, local checkout paths, or credentialed
remote URLs.
