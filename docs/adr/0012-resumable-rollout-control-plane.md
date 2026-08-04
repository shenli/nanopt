# ADR-0012: Simulate resumable rollout control before adding an accelerated backend

**Status:** Accepted for v0.4.

## Context

NanoPT v0.3 trains only on fresh, short, complete Agent RL groups. Long trajectories introduce
partial state, policy publication during execution, cache persistence, and stale or mixed-policy
data. Installing a distributed rollout engine before these identities are explicit would hide the
control-plane contract behind framework behavior and exceed the single-machine educational scope.

## Decision

v0.4 adds a deterministic CPU systems laboratory with:

- hash-bound model and world state captured only between complete tool actions;
- episode-boundary and action-boundary weight-synchronization experiments;
- external prefix-cache metadata keyed by exact prompt IDs and policy hash;
- explicit fresh, stale, and mixed-policy admission decisions;
- a hard declaration that simulated experience is never used for a model update;
- inspectable JSONL artifacts and one systems-engineer tutorial.

The lab does not install vLLM, allocate real KV blocks, execute a sandbox, or measure throughput.
An accelerated backend remains optional future work behind the frozen state and admission
contracts.

## Consequences

- Learners can observe the state/cache/freshness trade-off without a cluster or model download.
- Prefix reuse after a weight change is rejected by construction because the policy hash is part
  of cache identity.
- Episode-boundary synchronization exposes staleness; action-boundary synchronization exposes
  mixed-policy semantics and recomputation.
- The experiment cannot support performance or fault-tolerance claims.
- A future backend must preserve these records or document a stronger, tested replacement.
