# Milestone 10 completion report

Milestone 10 is complete with disclosed publication limitations. NanoPT v0.1.0 has a frozen lock,
model and tokenizer revision, numeric reference targets, release manifest, tested wheel and source
archive, strict documentation build, public-content audit, license inventory, fresh curriculum run,
and fresh reference-hardware pipeline and agent-environment runs.

The verified source tag targets candidate commit
`574582abd481545b724607f66e1ec014563cf95a`. Later documentation-only commits publish this report
and do not change the tagged implementation or its release archives.

## Reproduce the local release gate

On a clean checkout with Python 3.11 and `uv`:

```bash
./scripts/run_m10_release_gate.sh
```

The accepted run created a fresh locked development environment, passed formatting and linting,
strictly type-checked 87 source files, passed 400 tests with one explicit network-only skip,
validated 18 schemas and 12 YAML profiles, linted formulas in 84 Markdown files, built the strict
documentation site, and built both Python distributions. It then installed the wheel and its
declared dependencies into a second isolated environment and exercised `nanopt --version`,
`nanopt --help`, configuration resolution, and the read-only diagnostic command without downloading
a model.

## Release artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `nanopt-0.1.0-py3-none-any.whl` | 171,782 bytes | `663254fe9e4e9cf494b45f7ceff9654f95d1faa6ac47b1a96c452625549b5a7b` |
| `nanopt-0.1.0.tar.gz` | 454,286 bytes | `5d8d2241476f1cf9ef408723e7a438b8d2b3debb4b35c7b3e39471df8f45c933` |

The archive validator checked project name/version metadata, Apache-2.0 license inclusion, file
counts, expected filenames, and exclusion of private handoff material. The public-tree audit scanned
372 text files across 378 tracked files for personal paths, credential patterns, non-English
writing systems, private inputs, oversized files, and GitHub Actions workflows. The one redaction
test fixture is narrowly normalized before scanning; its tests prove the fake path and token do not
escape generated reports.

## Fresh curriculum and reference results

The fresh M9 gate executed all 20 unique local labs: 18 CPU labs and two systems simulations. The
reference host then reran the complete 15-stage Base → SFT → DPO → GRPO pipeline from a fresh locked
Linux environment:

| Checkpoint | Protected exact accuracy | Parse rate |
| --- | ---: | ---: |
| Base | 0.0% | 0.0% |
| SFT | 86.4% | 95.5% |
| DPO | 84.1% | 95.5% |
| GRPO | 88.6% | 95.5% |

The pipeline completed in 221.9 seconds with zero failed stages, repeated final generation exactly,
and peaked at 7,954,497,536 reserved bytes (7.41 GiB). The separate Docker protocol again solved
5/5 MiniSWE tasks with the oracle, replayed all five trajectories exactly, retained the deliberately
capped 0.3 base-model result, and passed the non-root/no-network/no-GPU/read-only-root/capability
isolation probes. The Whisper transcription service was stopped only during these GPU checks and
was confirmed active again afterward.

## Supply chain and publication decision

The model and tokenizer are pinned to Qwen revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, whose official metadata identifies Apache-2.0. NanoPT
references but does not redistribute those weights. The release gate captured metadata for 85
installed distributions; nine exact-version wheels omit a normalized license field, so their PyPI
license classifiers were reviewed and the metadata limitation is disclosed in the
[dependency audit](dependency-license-audit.md).

The source tag and private GitHub release pass. PyPI upload and a public announcement are deferred:
the repository is still private, PyPI credentials and a final trademark decision belong to the
owner, and v0.1 intentionally distributes no generated dataset or adapter. PyPI's official project
endpoint returned 404 for `nanopt` immediately before this decision, but that does not reserve the
name.

The reviewed aggregate evidence is
[`v0.1-release-574582a.json`](evidence/v0.1-release-574582a.json). Full clean-run records are attached
to the private GitHub release rather than committed with machine-specific transient data.
