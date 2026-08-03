# Dependency and model license audit

## Scope

This is a release-engineering inventory, not legal advice. NanoPT does not vendor its Python
dependencies, Qwen weights, generated datasets, or trained adapters in the v0.1 source and wheel
archives. Dependencies are installed from their own distributions and retain their own licenses.

The complete environment is fixed by `uv.lock`. The M10 release evidence records that file's hash
and the installed distribution/version/license metadata from a fresh locked environment.

## Direct runtime dependencies

| Dependency | Locked v0.1 version | Declared license metadata |
| --- | --- | --- |
| NumPy | 2.4.6 / 2.5.1 platform variants | BSD-3-Clause plus bundled permissive components |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause |
| PEFT | 0.20.0 | Apache-2.0 |
| Pydantic | 2.13.4 | MIT |
| PyYAML | 6.0.3 | MIT |
| Rich | 14.3.4 | MIT |
| safetensors | 0.8.0 | Apache-2.0 |
| PyTorch | 2.7.1 / 2.7.1+cu126 platform variants | BSD-3-Clause |
| Transformers | 5.14.1 | Apache-2.0 |
| Typer | 0.27.0 | MIT |

The development and documentation extras are also locked. Their direct dependencies declare
Apache-2.0, BSD, MIT, MPL-2.0, PSF-2.0, ISC, or similarly permissive combinations. Some wheels omit
a normalized `License-Expression`; for those, M10 reviewed the exact-version PyPI license
classifier and records the metadata limitation rather than interpreting an empty field as a
license failure.

## Model and container inputs

- `Qwen/Qwen3-0.6B-Base` is referenced, not redistributed. The official model metadata for pinned
  revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` identifies Apache-2.0.
- The MiniSWE backend references a digest-pinned official Python 3.11 slim image. NanoPT does not
  repackage that image in its Python artifacts.
- The synthetic arithmetic dataset has a separate [dataset card](../data/dataset-card.md). No bulk
  dataset or adapter is attached to v0.1.

## Release checks

The release gate verifies NanoPT's Apache-2.0 `LICENSE`, wheel metadata, model pin, archive contents,
tracked-file inventory, and dependency metadata. A future dependency, model, dataset, or bundled
binary requires a new audit; this document is not a blanket approval for later versions.
