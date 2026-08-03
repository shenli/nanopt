# Reading modern post-training systems with a fixed template

## Learning objectives

After this guide, you should be able to:

- compare open reports across stages, data, rewards, optimization, evaluation, and infrastructure;
- distinguish paper-supported facts from NanoPT design choices;
- mark undisclosed production details unknown;
- map industrial mechanisms to the small executable course without claiming equivalence.

## Comparison template

Use the same fields for every report: base model, stage sequence, cold-start/SFT data, preference
stage, RL algorithm, value model, rollout granularity, rewards/verifiers, KL/length controls, task
synthesis, tools/environment, evaluation/regressions, distillation, rollout/training infrastructure,
sandbox persistence, released code/data, and explicitly undisclosed details.

## Source-grounded comparison

| System | Public stage emphasis | NanoPT connection | Important unknown/omission |
| --- | --- | --- | --- |
| [Tülu 3](https://arxiv.org/abs/2411.15124) | Open SFT, DPO, RLVR recipe, data/evaluation | Closest staged open-pipeline reading companion | NanoPT's tiny data and single GPU do not reproduce its scale/mixtures |
| [Llama 3](https://arxiv.org/abs/2407.21783) | Large assistant post-training, synthetic/preference data, safety | Contrast for data mixtures and regression evaluation | Full production data, infrastructure, and operational details are not reproducible from the report |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | Math continuation and original GRPO presentation | Group-relative advantage/objective source | NanoPT does not reproduce the 7B model, 120B-token math corpus, or benchmark claims |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | RL reasoning behaviors, cold-start and multi-stage consolidation | Contrast for RLVR and behavior constraints | Exact proprietary data/infrastructure details not disclosed should remain unknown |
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | Long-context RL, policy optimization, long-to-short | Partial-rollout and long-tail systems reading | NanoPT's scheduler is a counter simulation, not the reported infrastructure |
| [Kimi K3](https://arxiv.org/abs/2607.24653) | Frontier multimodal/agent post-training and systems co-design | Current industrial contrast for stateful environments and rollout systems | NanoPT does not reproduce its model, scale, training data, or infrastructure |
| [HybridFlow/veRL](https://arxiv.org/abs/2409.19256) | Distributed RL dataflow and worker placement | Maps actor/reference/reward/rollout roles | v0.1 intentionally excludes its distributed runtime |

## How to read a claim

Label each note as one of:

- **Source-supported fact:** directly described by a primary paper/report/repository.
- **NanoPT decision:** a local implementation choice such as population standard deviation or
  synchronous rollout.
- **Measured NanoPT result:** bound to committed evidence and a pinned revision.
- **Hypothesis:** a proposed explanation or extension needing an experiment.
- **Unknown:** not disclosed or not verified.

Do not transfer a result across categories. A paper's large-scale benchmark result does not validate
NanoPT; NanoPT's exact CPU test does not prove cluster behavior.

## Reading sequence

1. Read each abstract/introduction for the claimed problem.
2. Locate the exact stage/data/objective sections.
3. Fill the template with page/section anchors and `unknown` entries.
4. Inspect released code/configs for defaults not fixed by the paper.
5. Compare to NanoPT's named config and tested function.
6. Finish with evaluation, ablations, failures, and disclosures—not just headline metrics.

## Exercise

Choose Tülu 3 and DeepSeek-R1. Fill every template field using only their reports and official
repositories. Highlight three fields where one report is more reproducible and leave unsupported
details unknown.
