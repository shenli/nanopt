# CLI reference

NanoPT exposes only commands backed by implemented, tested behavior:

```text
nanopt doctor
nanopt config resolve
nanopt data generate
nanopt calibrate --mode load|eval
nanopt eval run
nanopt report build RUN_DIR
nanopt artifacts inspect RUN_DIR
```

`nanopt data generate` writes the deterministic arithmetic task JSONL and split manifest. It
refuses to append to a non-empty output, preventing two dataset versions from being mixed.

`nanopt eval run` creates a run directory before loading the model, appends each example before
aggregation, and builds Markdown/HTML reports. Use `--mode deterministic` for one greedy completion
per task or `--mode sampled` for the profile's fixed sample count and seed schedule. The task JSONL
must have a sibling `dataset_manifest.json`; counts and canonical hashes are verified before model
loading.

`nanopt calibrate --mode load` exercises the exact model-loader path. `--mode eval` requires a task
JSONL and uses an explicit small limit; its manifest labels the result non-representative.

`nanopt report build RUN_DIR` is offline and model-free. It rebuilds all headline aggregates from
`samples.jsonl`. Run `uv run nanopt COMMAND --help` for the complete typed option surface.

Training, pipeline, and agent commands are added with their later vertical slices. The CLI does not
advertise placeholders that silently do nothing.
