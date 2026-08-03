# Security Policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for `shenli/nanopt`. Do not open a
public issue for a suspected secret exposure, sandbox escape, hidden-verifier leak, path traversal,
or dependency compromise. A maintainer will acknowledge a report and coordinate disclosure after a
fix is available.

## Supported versions

The `0.1.x` release line and the latest commit on the default branch receive security fixes. Alpha
releases do not yet have a fixed end-of-support date; any support-policy change will be recorded in
this file and in release notes.

## Threat model

Model output, tool arguments, generated patches, and task workspaces are untrusted. Future agent
tasks will use allow-listed structured tools, no network or GPU, non-root execution, resource
limits, and a verifier workspace separate from model-visible public tests. Containers reduce risk;
they are not a perfect boundary against hostile kernel-level attacks.

Never run unreviewed pull-request code on a persistent self-hosted GPU runner, mount credentials or
the Docker socket into an agent sandbox, or publish hidden verifier content in artifacts.
