# Milestone 6 completion report

Milestone 6 is complete on the CPU and clean reference-GPU tiers. The accepted bundle used commit
`8a35e467393b69717f27878354d720d2f4060086` and DPO parent adapter
`b5bb5a8b7421f283a457533d6185e64fedc95d30dab536f082042833cd9036ab`.

## Delivered vertical slice

- deterministic prompt scheduling and fresh grouped sampling with $G \ge 2$;
- exact sampled IDs, action masks, behavior log probabilities, finish reasons, and rewards;
- strict parser/verifier-separated arithmetic reward and a five-case hacking suite;
- population group-relative advantages with explicit degenerate groups;
- direct stored-ID collation, current/old token ratios, sign-correct clipping, and two named
  normalization modes;
- optional exact stored-ID frozen-adapter scoring with direct or k3 KL estimators;
- synchronous LoRA training, metrics, trajectory/reward artifacts, schemas, CLI calibration,
  chapters, ADR, CPU lab, reference runner, and offline validator.

## Pilot and target freeze

A one-iteration calibration exercised generation, verification, advantages, backward, clipping,
and optimizer phases. Non-representative 4- and 12-iteration pilots then established recipe
behavior before targets were frozen in `configs/reference_targets.yaml` at commit `b9feb76`.

The 12-iteration pilot improved both compositional and range protected splits. Frozen targets
required at least one primary split improvement, no more than a five-point overall accuracy/parse
regression, rollout parser/correctness rates of at least 15%, no more than 50% degenerate groups,
mean clipping below 10%, ratio p95 below 1.5, and zero hacking correctness credit.

## Accepted reference result

The run generated 12 trajectories and 48 exact-token completions:

| Rollout/update metric | Result |
| --- | ---: |
| Mean reward | 0.2771 |
| Exact correctness | 25.0% |
| Parser success | 27.1% |
| Degenerate groups | 33.3% |
| Mean clip fraction | 0.18% |
| Maximum per-iteration ratio p95 | 1.094 |
| Peak reserved memory | 7,948,206,080 bytes (7.40 GiB) |

All five reward-hacking cases received zero correctness credit. The training path used
`group_zscore`, `token_mean`, clip epsilon 0.2, one update epoch, and `kl_beta=0`; there was no
explicit frozen-reference KL penalty.

Protected deterministic accuracy improved from DPO's 37/44 (84.1%) to GRPO's 39/44 (88.6%) while
parse rate stayed 42/44 (95.5%). Compositional accuracy improved from 9/12 to 10/12 and range from
10/12 to 11/12; IID and format-attack results were unchanged.

## Rejected evidence attempt

The first frozen-target run at `f83f08a` completed all GPU work and met every target, but its
offline validator reused an object-only JSON helper for the intentionally array-shaped
`reward_hacking.json`. That evidence attempt was rejected. The reader was fixed, the earlier bundle
was dry-validated to find any remaining validator defects, and the full protocol was rerun cleanly
at `8a35e46`.

## Evidence

The compact reviewed evidence is
[`m6-reference-grpo-8a35e46.json`](evidence/m6-reference-grpo-8a35e46.json). The full ignored bundle
remains on the reference host at `artifacts/tmp/m6-reference-20260803-164702`.

This proves M6, not M7 hardware validation. The official pipeline must still rebuild every stage
from one clean checkout without source edits and produce parent/child manifests, wall times,
checksums, and a comparison report.
