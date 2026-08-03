# Exact rollouts and verifiable rewards

## Learning objectives

After this chapter, you should be able to:

- explain why policy-gradient training cannot decode and re-tokenize a rollout;
- trace sampled IDs, old log probabilities, masks, finish reasons, and rewards in one trajectory;
- distinguish parser invalidity, incorrect verification, and verifier errors;
- identify degenerate equal-reward groups before optimization;
- audit the arithmetic reward against fixed hacking attempts.

## The sampled tokens are the actions

In language-model RL, each generated token ID is an action. The old-policy log probability stored
for that action must refer to the exact categorical draw that produced it. Decoded text is a lossy
view: tokenization need not be unique, whitespace can normalize, and special tokens may disappear.

NanoPT therefore stores, for every completion:

```text
token_ids          [completion]
action_mask        [completion]
old_logprobs       [completion]
reference_logprobs [completion] or absent
finish_reason      scalar label
reward/components  scalar and named values
advantage          response-level scalar
decoded_text       parser/reviewer view only
```

[`generate_grouped_trajectory`](https://github.com/shenli/nanopt/blob/main/src/nanopt/grpo/rollout.py)
uses the explicit autoregressive sampler at temperature 1 and top-p 1. The behavior distribution is
then the unmodified policy distribution, so the stored behavior log probability is the exact
old-policy value used by GRPO.

The arithmetic answer closing tag is an environment protocol stop, just as EOS is a model stop.
The closing-tag token IDs remain part of the completion and action mask. The record labels the
finish reason `protocol_stop`; it does not pretend that the model emitted generic EOS.

## Decode once, only for reward

Decoded text crosses into the strict parser and exact AST verifier. The reward is

$$
r = 1.0 r_{\mathrm{correct}} + 0.1 r_{\mathrm{format}}
    - \lambda_{\mathrm{length}} p_{\mathrm{length}}.
$$

The first reference run uses $\lambda_{\mathrm{length}}=0$. The named length component is still
recorded, which makes later changes visible.

[`arithmetic_rlvr_reward`](https://github.com/shenli/nanopt/blob/main/src/nanopt/grpo/reward.py)
keeps three states separate:

- parser valid or invalid;
- verifier correct, incorrect, or not run;
- verifier contract error.

A malformed response receives neither format nor correctness reward. A well-formed wrong answer
receives format reward but no correctness reward. A trusted-task contradiction is an error, not an
ordinary negative sample.

## Reward-hacking suite

Before rollout training, NanoPT runs fixed attacks including multiple answers, trailing content,
case-shifted tags, answer-tag attributes, and a canonical wrong value. Every case must receive zero
correctness credit. The raw attack strings and component rewards are saved in
`reward_hacking.json`.

This suite does not prove the verifier can never be hacked. It proves known contract boundaries
still behave exactly as tested before the policy is optimized against them.

## Group rewards and degenerate data

Completions are sampled in groups with $G \ge 2$. Advantages are calculated only after all group
rewards exist. If every reward is equal, population standard deviation is zero and every advantage
is exactly zero. The trajectory remains valuable evidence about rollout behavior, but it supplies
no policy-gradient signal. NanoPT logs the degenerate-group fraction instead of silently dropping
or inventing signal.

## CPU inspection

Run:

```bash
uv run python labs/09_exact_rlvr_trajectory.py
```

The lab serializes exact token coordinates, reloads them, and collates them into causal prediction
coordinates without consulting decoded text.

## Common mistakes

- Decoding and re-tokenizing completions before the update.
- Storing sequence-summed old log probabilities when the ratio is token-level.
- Excluding the terminating sampled token from the action mask.
- Calling parser failure an incorrect verified answer.
- Granting correctness credit before independently checking the trusted AST.
- Hiding all-equal groups by dividing through an epsilon and reporting nonzero signal.

Production rollout systems batch and cache aggressively, but they must preserve the same action
identity and reward contract. Throughput is not permission to change which policy distribution or
token coordinate a stored log probability describes.

