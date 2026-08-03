# Milestone 8 completion report

Milestone 8 is complete. NanoPT now includes an original five-task MiniSWE suite, strict structured
tools, deterministic reset/replay, isolated public and hidden verification, a local teaching
backend, and a hardened Docker reference backend. The environment records model action tokens and
evaluates policies; it does not train or update them.

## Reproduce the protocol

On a clean Linux checkout with the pinned model and Docker image already cached:

```bash
./scripts/run_m8_reference_agent.sh
```

The script creates a fresh Python 3.11 environment from `uv.lock`, runs strict hardware diagnosis,
evaluates every oracle task in Docker, retains a capped real-model baseline, runs adversarial
sandbox probes, hashes 38 retained files, and invokes the offline validator. The accepted run used
commit `9e3daa1b29c34616e5c5b9c32926ad322ecce909`.

## Task and replay results

| Policy | Scope | Solved | Mean score | Steps | Wall time |
| --- | --- | ---: | ---: | ---: | ---: |
| Scripted oracle | 5-task representative suite | 5/5 | 1.000 | 15 | 15.29 s |
| Qwen base | 1 task, 2-turn cap | 0/1 | 0.300 | 2 | 19.92 s |

The oracle submitted each reviewed diff through the same `apply_patch`, `run_tests`, and `finish`
protocol available to a model. Public and hidden tests passed on all five tasks. Replaying all 15
retained response objects from new immutable resets produced five exact semantic matches.

The Qwen result is deliberately labeled non-representative. Both 256-token responses failed strict
JSON parsing, consumed their turns, and were retained with exact sampled token IDs. The unchanged
snapshot passed 2/2 public tests and 1/2 hidden tests; two invalid-action penalties reduced the raw
0.5 hidden result to 0.3. This unsuccessful baseline is evidence about the unadapted policy, not a
reason to weaken the protocol.

## Isolation and security

The reference backend used the digest-pinned Python 3.11 image and Docker 29.5.0. Runtime probes
confirmed:

- numeric non-root UID/GID 65532;
- outbound network blocked;
- no visible GPU devices or Docker socket;
- zero effective Linux capabilities and `NoNewPrivs=1`;
- read-only container root with only the workspace writable;
- configured CPU, memory/swap, PID, wall-time, and output limits.

Public tests and hidden tests execute in different disposable copies of the submission. Hidden-test
source, paths, and process output are absent from model observations and public trajectories. Unit
tests separately cover path traversal, symbolic-link escape, protected-test edits, atomic patch
failure, bounded output, and timeout behavior.

The first candidate run exposed that Docker 29 rejects a bare `rw` field in `--mount`; bind mounts
are writable by default. The failed attempt was not accepted. Commit `9e3daa1` removed the invalid
field, added a regression assertion, passed the complete local gate, and then passed the fresh
reference protocol.

## Scope and evidence

The full ignored bundle remains on the reference host. The reviewed compact evidence is
[`m8-reference-agent-9e3daa1.json`](evidence/m8-reference-agent-9e3daa1.json). It contains hashes
and aggregate security/results data without machine-specific paths, hidden-test contents, model
responses, or weights.

Docker is a risk-reduction layer, not a claim of safety against a kernel or daemon compromise. The
fake backend remains restricted to trusted tests and CPU teaching labs. Agent SFT and Agent RL are
deferred to later releases; Milestone 9 completes the remaining curriculum and troubleshooting
material.
