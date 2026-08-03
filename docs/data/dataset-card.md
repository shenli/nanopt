# Synthetic arithmetic dataset card

## Learning objectives

After reading this card, you should be able to explain what NanoPT generates, which splits are
protected, why the data is intentionally narrow, and what a separate dataset publication would
need to disclose.

## Summary

NanoPT generates exact-answer arithmetic tasks locally from versioned project code. The v0.1
repository does not ship a bulk dataset: the documented commands construct it deterministically
from a configuration and seed. This keeps the teaching path inspectable and avoids importing an
unreviewed instruction-data license or hidden benchmark content.

Each record contains a task identifier, task family, prompt, canonical expression, trusted answer,
generator version, seed-derived lineage, and split. The parser accepts only the documented answer
protocol; the verifier independently evaluates the restricted arithmetic expression rather than
trusting a stored model response.

## Intended uses

- Teaching completion-only SFT, controlled preference construction, DPO, and verifiable RL.
- Hand-checking data fingerprints, protected splits, parser behavior, and regression reports.
- Small deterministic smoke and reference-hardware experiments.

The data is not intended to measure broad mathematical reasoning, instruction following, safety,
coding, factual knowledge, or production-agent quality.

## Generation and splits

`nanopt data generate` uses `src/nanopt/data/arithmetic.py` and the checked-in configuration. The
same generator version, configuration, and seed produce the same content fingerprint. Canonical
task hashes prevent overlap between training and protected evaluation splits. Preference examples
are derived by deterministic, labeled rejection transformations; they are not human preference
annotations.

## Sensitive data and provenance

The generator uses no user sessions, personal data, web scrape, proprietary corpus, or external
dataset. Generated examples consist only of project-created templates, integers, operators, and
answers. MiniSWE task fixtures are a separate original suite and are not part of this dataset.

## License and distribution

The generator code and checked-in examples are provided under NanoPT's Apache-2.0 license. NanoPT
v0.1 does not attach a bulk generated dataset to the release. Anyone publishing generated output
separately should preserve its generator/configuration fingerprint, include this card, record the
exact NanoPT revision, and make an explicit license and hosting decision for that publication.

## Limitations

Synthetic exact answers make verification unusually cheap and reliable. Real post-training data
contains ambiguous instructions, subjective preferences, contamination, personal information,
copyright questions, and imperfect rewards. Results on this dataset must not be presented as a
general model-quality claim.

## Reproduce the local sample

```bash
uv run nanopt data generate \
  --output artifacts/tmp/tasks.jsonl \
  --manifest artifacts/tmp/dataset_manifest.json
```

Inspect the manifest before using the records. It is the source of truth for generator version,
configuration, split counts, and fingerprints.
