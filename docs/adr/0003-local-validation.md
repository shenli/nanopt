# ADR-0003: Use local validation instead of GitHub Actions

- Status: Accepted
- Date: 2026-08-02
- Owners: NanoPT maintainers

## Context

Hosted CI consumes repository Actions budget. NanoPT's current maintainer prefers explicit local
validation while the project is small and pre-alpha.

## Decision

NanoPT does not use GitHub Actions. All CPU, type, lint, schema, package, and strict documentation
checks run locally before milestone commits. GPU/reference checks run only on an explicitly selected
local machine. Completion reports record the commands and results.

Documentation may be deployed from a maintainer machine to the `gh-pages` branch with
`mkdocs gh-deploy`; this does not require an Actions workflow.

## Alternatives considered

- Hosted CPU/docs workflows were removed to avoid Actions usage.
- A self-hosted Actions runner was rejected because it adds security and maintenance burden.

## Consequences

Public pull requests do not receive an automatic green check. Maintainers must run and report the
local gate consistently. Hosted CI may return later only through a superseding ADR and explicit
owner approval.

## Validation

The documented command sequence must pass in a fresh locked environment before every milestone
commit or release.
