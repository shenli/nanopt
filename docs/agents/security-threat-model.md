# Agent sandbox security and threat model

## Learning objectives

After this chapter, you should be able to:

- identify what NanoPT treats as trusted and untrusted;
- explain the separate roles of structured tools, workspace checks, and Docker isolation;
- map each reference security flag to a concrete risk reduction;
- state what the M8 sandbox does not promise;
- run the security lab without exposing a GPU or network to task code.

## Trust boundary

The repository owner, reviewed task cards, hidden verifier, pinned container image, host kernel,
Docker daemon, and NanoPT implementation are trusted. Model responses and files modified during an
episode are untrusted. The tiny checked-in task snapshots are auditable inputs, but their executed
code is still contained by the reference backend.

Defense is layered:

| Layer | Enforced property |
| --- | --- |
| Typed action protocol | Exactly one allow-listed JSON tool; no arbitrary shell action |
| Safe workspace | Relative paths only, no symlinks, bounded reads/searches, editable globs |
| Patch boundary | Existing UTF-8 files only, 64 KiB limit, protected files rejected, atomic replace |
| Process boundary | Trusted argv, timeout, process-group termination, bounded sanitized output |
| Docker boundary | Non-root, no network/GPU/capabilities, read-only root, resource limits |
| Hidden verifier | Fresh copied submission, tests injected separately, no hidden source/output returned |
| Audit records | Budget transitions, hashes, violations, termination, and scores retained |

No layer is a substitute for the others. A strict JSON schema does not contain buggy submitted
Python, and a container does not make ambiguous path handling safe.

## Reference Docker contract

[`DockerSandboxBackend`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/sandbox/docker.py)
constructs argv directly. It requires a SHA-256-pinned official Python image already present on the
host and applies:

- `--network none`;
- numeric UID/GID `65532:65532`;
- `--cap-drop ALL` and `no-new-privileges`;
- a read-only container root and small `noexec,nosuid,nodev` `/tmp`;
- one writable bind mount at `/workspace`;
- fixed memory, swap, PID, CPU, wall-time, and output limits;
- no GPU flags and no Docker socket mount.

The M8 reference security probe checks the effective UID/GID, zero effective capabilities,
no-new-privileges bit, blocked outbound connection, blocked root-filesystem write, writable
workspace, and absence of GPU devices and the Docker socket.

## Explicit non-goals

This is a local educational sandbox, not a multi-tenant hostile-code service. It does not claim to
defend against a Docker or kernel escape, microarchitectural side channels, denial of service beyond
the configured local limits, a malicious trusted image/task card, or a compromised Docker daemon.
Run reference evaluations only on a machine where the operator accepts those residual risks.

The fake backend offers bounded output and timeouts for trusted tests but runs on the host. Never use
it to evaluate code from an untrusted model or third party.

## Security lab

First run the CPU environment/replay lab:

```bash
uv run python labs/10_mini_swe_environment.py
```

On a Linux Docker host with the pinned image already installed, run the reference probes as part of:

```bash
./scripts/run_m8_reference_agent.sh
```

Read `security_probes.json` before accepting the run. A green aggregate status without the concrete
probe values is insufficient evidence.

## Exercises

1. Change a lab action path to `../outside.py` and inspect the violation and budget charge.
2. Attempt to patch `tests/test_*.py`; explain why both pre-execution rejection and final hash checks
   are useful.
3. Lower the output limit and create a trusted fixture that prints indefinitely; verify truncation
   and timeout are distinct fields.
4. Explain why public and hidden tests run from different workspaces even though both commands are
   trusted.
