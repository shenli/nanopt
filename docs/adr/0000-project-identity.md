# ADR-0000: Project identity and initial license

- Status: Accepted
- Date: 2026-08-02
- Owners: NanoPT maintainers

## Context

M0 requires one repository identity, a distribution-name decision, and a license selection before
the foundation publishes package metadata.

## Decision

- The repository slug is `shenli/nanopt`, matching this checkout's configured origin.
- The distribution, import package, and CLI are all named `nanopt`. An authoritative PyPI JSON
  lookup for `nanopt` returned HTTP 404 on 2026-08-02. Availability must be checked again
  immediately before the first upload because a lookup does not reserve the name.
- Apache-2.0 is selected, matching the license already committed in the repository.
- If the distribution name becomes unavailable, the distribution may change to `nanopt-llm` while
  the import and CLI remain `nanopt`; that change requires a superseding ADR.

## Alternatives considered

- Deferring all package metadata was rejected because it prevents a clean M1 build and lockfile.
- Using different import and CLI names was rejected because both are fixed project constraints.

## Consequences

Wheel metadata, documentation URLs, and the CLI now use one concrete identity. This ADR does not
publish or reserve a PyPI project and does not establish trademark availability.

## Validation

The local release gate builds the `nanopt` wheel, imports `nanopt`, and runs the `nanopt` entry point.
The publication checklist blocks upload until the name and metadata are rechecked.
