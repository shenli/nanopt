# Production data flywheels without evaluation leakage

## Learning objectives

After this chapter, you should be able to:

- distinguish task discovery from direct training-data ingestion;
- explain consent, privacy, delayed outcomes, and environment reconstruction boundaries;
- keep fixed evaluation examples out of adaptation loops;
- separate replaying a session from generating a fresh-policy rollout;
- design shadow, canary, and rollback stages around a synthetic candidate task.

## Discovery is not training

A production failure can suggest a useful task family, but the raw session may contain private
content, lack consent, depend on unavailable state, or have an outcome known only later. NanoPT's
teaching flow is:

```text
delayed synthetic signals
→ consent/privacy/outcome filters
→ aggregated failure pattern
→ human-authored licensed task and verifier
→ fixed evaluation or training pool (never both)
→ fresh-policy rollout
→ shadow → canary → rollback decision
```

The candidate queue records task family, failure code, count, and a digest of synthetic source IDs.
It does not retain raw conversation content.

## CPU systems lab

```bash
uv run python labs/16_production_flywheel.py
```

[`build_task_candidates`](https://github.com/shenli/nanopt/blob/main/src/nanopt/systems/flywheel.py)
rejects signals without research consent, sensitive signals, outcomes that arrived too early,
successful sessions, and fixed-evaluation membership. It aggregates accepted failures for human
review. The function does not generate executable tasks or update a model.

## Delayed outcomes and fresh policy

A user retry, test result, or later correction may be more informative than an immediate thumbs-up.
Joining delayed outcomes requires stable privacy-aware identifiers and a retention policy. Blindly
replaying an old session measures the old policy/environment path. To evaluate a new policy, rebuild
the authorized environment and roll out the new policy while keeping the fixed comparison contract.

## Deployment evaluation

- **Fixed evaluation:** versioned, isolated, never adapted on.
- **Shadow:** new policy runs without affecting users; compare outcomes and resource use.
- **Canary:** small authorized traffic share with explicit stop conditions.
- **Rollback:** restore a known checkpoint/config when safety, quality, or operations regress.

NanoPT does not implement live collection or deployment. This chapter is a design simulation so the
course can discuss feedback loops without handling real personal data.

## Common mistakes and scale mapping

- Treating terms-of-service access as research consent.
- Moving a failed evaluation example into training while keeping it in the headline test set.
- Hashing raw text and claiming it is anonymized.
- Using old-policy session success as proof a new policy will succeed.
- Launching a canary without a rollback artifact and threshold.

Production systems need legal/privacy review, deletion workflows, access controls, audit logging,
drift monitoring, and incident response. The small simulation only makes filtering and lineage
decisions executable.

## Exercises

1. Add a duplicate synthetic signal and explain why aggregation is preferable to raw retention.
2. Design a split rule that prevents a task family from leaking between adaptation and fixed eval.
3. Write canary rollback conditions for correctness, latency, and policy violations.
