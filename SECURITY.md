# Security Policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for `shenli/nanopt` when it is
available. If that interface is unavailable, contact the maintainer privately using the contact
details on the [`shenli` GitHub profile](https://github.com/shenli). Do not open a public issue for
a suspected secret exposure, sandbox escape, hidden-verifier leak, path traversal, or dependency
compromise. A maintainer will acknowledge a report and coordinate disclosure after a fix is
available.

## Supported versions

The latest tagged release (currently the `0.3.x` line) and the latest commit on the default branch
receive security fixes. Earlier alpha lines are historical and do not receive backports. Alpha
releases do not yet have a fixed end-of-support date; any support-policy change will be recorded in
this file and in release notes.

## Threat model

Model output, tool arguments, generated patches, and task workspaces are untrusted. The reference
agent uses allow-listed structured tools, no sandbox network or GPU, non-root execution, resource
limits, and a verifier workspace separate from model-visible public tests. Containers reduce risk;
they are not a perfect boundary against hostile kernel-level attacks.

Never run unreviewed pull-request code on a persistent self-hosted GPU runner, mount credentials or
the Docker socket into an agent sandbox, or publish hidden verifier content in artifacts.
