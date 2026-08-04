# Troubleshooting from observed NanoPT runs

## Learning objectives

After this guide, you should be able to:

- classify a failure as environment, data, protocol, numerical, memory, sandbox, or lineage related;
- preserve rejected evidence instead of editing around a failing gate;
- choose the smallest diagnostic command that tests the failed boundary;
- distinguish a real implementation defect from an honest model-quality result.

## Start with the boundary, not the symptom

Run `uv run nanopt doctor --json doctor.json`, then inspect the failing run's `run_manifest.json`,
`events.jsonl`, and example-first artifacts. Do not rerun with looser parsing, different data, or a
smaller hidden test until you understand which contract failed.

| Symptom | Likely boundary | First check |
| --- | --- | --- |
| Import/dependency mismatch | Locked environment | `uv sync --frozen --extra dev --extra docs` |
| CUDA unavailable or wrong GPU | Runtime/profile | `uv run nanopt doctor --strict-profile` |
| Valid-looking answer parses as invalid | Renderer/protocol stop | Inspect exact token IDs, decoded suffix, finish reason |
| Live/reference log probabilities differ | Numerical/cache contract | Compare exact input IDs, masks, dtype, forward layout |
| GRPO loss moves but reward does not | Reward/data freshness | Inspect groups, degenerate fraction, attack suite, checkpoint hash |
| Resume refuses a completed stage | Lineage | Re-hash child manifest and output checkpoint; do not bypass |
| Docker tests all fail immediately | Sandbox argv/image/daemon | Inspect `backend_details` and run security probes |
| Hidden verifier gives no text | Expected isolation | Use counts/status; do not expose hidden output for debugging |

## Observed lessons

### Trailing generation text

Early supervised fine-tuning pilots produced correct answers followed by extra output. The strict
parser rejected them.
The accepted fix was an exact token-stop contract that retained the closing answer tag—not a weaker
parser. When a format metric changes, inspect sampled IDs and finish reason before changing data.

### BF16 cache parity

An early DPO pilot cached chosen and rejected sequences separately while the live path concatenated them.
BF16 execution produced small differences. The accepted path computed a complete zero-error parity
cache under one consistent forward layout. “Close enough” was not substituted for the declared
exact-cache contract.

### Reward-hacking evidence

The GRPO evidence reader briefly treated an object-like structure too permissively. The validator was
hardened and the attack lesson retained. Never allow evaluation artifacts to become a policy-facing
answer source.

### Docker mount syntax

The first agent-environment candidate used a bare `rw` token in Docker's `--mount` value. Docker 29 rejected it;
bind mounts are writable by default. The failed run was retained, the argv builder gained a
regression assertion, and a fresh clean candidate passed. A command reporting “episode completed”
is not sufficient—check oracle score and sandbox probe status.

## Memory failures

Record peak allocated/reserved memory by phase. Reduce only an explicit knob such as microbatch,
sequence length, rollout group, or activation checkpointing, and relabel the run when it no longer
matches the reference profile. Clear dead model objects between stages; do not claim a leak without
measuring allocator state.

## Model quality versus system correctness

The base-model agent evaluation emitted two invalid JSON actions and scored 0.3. That is a valid
unsuccessful baseline, not an environment failure: exact tokens, budget charges, hidden isolation,
and final verification were retained. Conversely, a scripted oracle that cannot solve every task
is a system or task failure even if a model sometimes succeeds.

## Escalation checklist

1. Save the exact command, revision, resolved config, and first failing artifact.
2. Reproduce the smallest deterministic boundary test.
3. Compare against the last accepted evidence hash.
4. Add a regression test before changing the implementation.
5. Rerun the complete local gate and the relevant reference protocol.
6. Document the rejected attempt and narrowed claim.
