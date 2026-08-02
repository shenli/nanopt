# ADR-0001: Keep foundation profiles typed and namespaced

- Status: Accepted
- Date: 2026-08-02
- Owners: NanoPT maintainers

## Context

Hardware, model, and experiment profiles reuse field names such as `id`, `status`, and `adapter`.
Flattening them through a generic deep merge could silently replace unrelated settings. Installed
CLI use must also work when the current directory is not the source checkout.

## Decision

Resolved documents retain separate `hardware`, `model`, and `experiment` namespaces. Every profile
is validated by an extra-forbidding Pydantic model. Dotted CLI overrides target scalar leaves;
unprefixed paths mean experiment fields. Canonical profiles live in `configs/` and Hatch includes
the same directory in wheels as `nanopt/_builtin_configs`.

## Alternatives considered

- A flat deep merge was rejected because profile identity and adapter fields collide.
- Untyped nested dictionaries were rejected because they cannot enforce unknown-key rejection.
- Profiles available only from the checkout were rejected because installed CLI behavior would
  depend on the caller's working directory.

## Consequences

The configuration models are verbose and objective-specific, but configuration errors are local and
readable. Adding a new profile shape requires extending the union and its tests.

## Validation

All canonical profiles must validate. Tests cover unknown keys, precedence, list/scalar rejection,
type mismatches, recipe-stage overrides, stable YAML, and wheel-bundled profile resolution.
