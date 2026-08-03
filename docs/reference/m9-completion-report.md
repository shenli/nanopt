# Milestone 9 completion report

Milestone 9 is complete. NanoPT now has one prerequisite chapter, all 20 numbered v0.1 chapters,
20 executable local labs, primary-source reading guides, troubleshooting based on observed runs, a
glossary, and contribution paths for algorithms, tasks, and hardware profiles.

## Reproduce the curriculum gate

From a clean checkout with Python 3.11 and `uv` available:

```bash
./scripts/run_m9_curriculum_gate.sh
```

The script creates a fresh environment from `uv.lock`, validates the curriculum schema and MkDocs
navigation, checks chapter learning objectives, verifies prior reference-evidence hashes, and runs
every unique CPU or systems-simulation lab. The accepted run used commit
`3d721c2bf596304b5cfa515f52d81cfde2a30376`.

## Curriculum and lab results

| Check | Result |
| --- | ---: |
| Prerequisite chapters | 1 |
| Numbered chapters | 20 |
| Declared lab uses | 29 |
| Unique local labs executed | 20/20 |
| CPU labs | 18 |
| Systems simulations | 2 |
| Prior reference declarations | 7 |
| Distinct prior evidence files | 6 |

The local labs completed in 28.8 seconds. Several chapters intentionally reuse a lab when the same
mechanism supports more than one lesson; the gate executes each unique command once and records its
output hash. The complete local quality gate also passed 398 tests with one explicitly skipped
network test, strict type checking across 87 source files, 17 schema validations, formula linting
across 82 Markdown files, a strict documentation build, and source/wheel builds.

## Systems lessons

The rollout-scheduler simulation compares two deterministic responses to a weight update. Letting
in-flight work finish completes in 8 ticks, preserves all work, and accepts one stale rollout.
Restarting partial work completes in 14 ticks, avoids stale rollouts, and discards six units of
work. This is a small teaching model of the throughput/freshness tradeoff, not a benchmark of a
distributed rollout engine.

The production-flywheel simulation accepts only consented, non-sensitive, successful sessions with
an unresolved failure and excludes fixed-evaluation material. It aggregates failure patterns
without retaining raw session content. This demonstrates an auditable filter boundary; it is not a
claim that real production data is safe merely because it passed these example predicates.

## Reading and extension scope

The reading guide uses a fixed template to distinguish source-supported claims, NanoPT design
choices, simplifications, and open hypotheses. It covers the objectives and systems represented by
the Tülu 3, Llama 3, DeepSeekMath, DeepSeek-R1, Kimi k1.5, HybridFlow/veRL, and Kimi K3 primary
sources. Separate extension guides define the tests and evidence required for new algorithms,
MiniSWE tasks, and hardware profiles.

## Evidence and limitations

The full execution record is intentionally ignored because it contains transient environment
details. The reviewed compact evidence is
[`m9-curriculum-3d721c2.json`](evidence/m9-curriculum-3d721c2.json). It binds the curriculum,
chapter files, lab files, local execution records, and prior reference evidence by SHA-256.

M9 did not repeat GPU training. Instead, its validator checked the status and exact hash of the
accepted M3–M8 evidence that the chapters cite. The systems exercises are deterministic local
simulations and do not establish multi-node performance or production privacy. Milestone 10 is the
v0.1 release audit: freeze the release inputs, rerun the required gates, audit public content and
licenses, build release artifacts, and tag only the verified candidate.
