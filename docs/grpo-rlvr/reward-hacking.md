# Reward hacking and verifier design

## Learning objectives

After this chapter, you should be able to:

- separate parsing, trusted verification, and scalar reward construction;
- explain how an exact verifier can still encode a flawed contract;
- run a fixed adversarial suite before optimization;
- distinguish malformed, wrong, and verifier-error outcomes;
- design public feedback without exposing protected answers.

## Three boundaries, not one score

NanoPT first parses a response into a narrow candidate answer, then compares it with a target derived
from the trusted arithmetic AST, then combines named reward components. A response can therefore be:

- malformed, so verification does not run;
- well formed but incorrect;
- correct under the trusted task;
- impossible to judge because the trusted task/verifier contract is inconsistent.

Collapsing these states into one zero hides whether the policy failed or the evaluation system did.
[`arithmetic_rlvr_reward`](https://github.com/shenli/nanopt/blob/main/src/nanopt/grpo/reward.py)
retains parser and verifier states alongside named reward components.

## Fixed attacks

The attack suite checks a canonical wrong value, multiple answer tags, trailing content,
case-shifted tags, and an answer-tag attribute. Run it before training:

```bash
uv run python labs/14_reward_hacking.py
```

Every attack must receive zero correctness credit. This is regression evidence for known boundaries,
not a proof that no unknown exploit exists. M6 preserved a rejected evidence-reader attempt because
the policy-facing reward must never learn answers from reference artifacts.

## Public and hidden verification

Public tests teach the contract and support debugging. Hidden tests reduce direct overfitting but
must remain isolated from observations, model-visible files, and tool output. A hidden test is not
automatically good: it can be brittle, leak through timing/errors, or reward one implementation
style instead of behavior. The M8 verifier uses separate disposable workspaces and returns hidden
counts only.

## Common mistakes and scale mapping

- Using the model's claimed answer as the trusted target.
- Awarding format credit for multiple ambiguous answers.
- Treating a verifier exception as an ordinary incorrect sample.
- Tuning repeatedly on a fixed hidden suite until it becomes public in practice.
- Adding a reward component without logging its unweighted value and coefficient.

Large RLVR systems use diverse task generators, adversarial evaluation, held-out verifiers, and
monitoring for reward/quality divergence. The same principle applies: optimize one boundary while
measuring independent boundaries.

## Reading and exercises

The [Tülu 3 report](https://arxiv.org/abs/2411.15124) is a primary open reference for staged
post-training with verifiable rewards. The [DeepSeek-R1 report](https://arxiv.org/abs/2501.12948)
discusses reasoning RL and observed behavior constraints at much larger scale.

1. Weaken the parser to accept trailing text and predict which fixed attack fails.
2. Design a hidden test that checks behavior without prescribing source structure.
3. Explain why a reward-model ensemble does not eliminate reward hacking.
