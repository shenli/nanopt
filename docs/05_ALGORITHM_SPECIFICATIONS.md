# Algorithm Specifications

This document defines the mathematical and implementation behavior required for NanoPT v0.1. The formulas are part of the software contract. Alternative variants may be added, but the reference recipe must identify exactly which variant it uses.

## 1. Notation

- $x$: prompt tokens.
- $y = (y_1, \ldots, y_T)$: generated or target completion tokens.
- $\pi_\theta$: trainable policy.
- $\pi_{\text{ref}}$: frozen reference policy.
- $\pi_{\text{old}}$: policy that generated a stored rollout.
- $m_t \in \{0,1\}$: completion/action-token mask.
- $r$: scalar outcome reward.
- $A$: advantage assigned to a response or token.
- $G$: number of sampled completions in one prompt group.
- $\epsilon$: numerical or clipping constant, depending on context.

A completion sequence log probability is:

$$
\log \pi_\theta(y \mid x)
=
\sum_{t=1}^{T} m_t
\log \pi_\theta(y_t \mid x, y_{<t}).
$$

Unless a specification explicitly says otherwise, DPO uses the **sum** over active completion tokens. Length-normalized values may be logged as diagnostics but must not silently replace the objective.

## 2. Causal token log probabilities

Given:

- `logits`: `[batch, sequence, vocabulary]`;
- `input_ids`: `[batch, sequence]`;
- `action_mask`: `[batch, sequence]`, aligned to token IDs;

compute:

```python
prediction_logits = logits[:, :-1, :]
target_ids = input_ids[:, 1:]
prediction_mask = action_mask[:, 1:]
log_probs = log_softmax(prediction_logits, dim=-1)
token_logps = gather(log_probs, target_ids)
```

The result has shape `[batch, sequence - 1]`.

Required functions:

```python
def causal_token_logps(
    logits: Tensor,
    input_ids: Tensor,
) -> Tensor: ...


def masked_sum(values: Tensor, mask: Tensor, dim: int) -> Tensor: ...


def masked_mean(values: Tensor, mask: Tensor, dim: int | tuple[int, ...]) -> Tensor: ...


def completion_sequence_logps(
    logits: Tensor,
    input_ids: Tensor,
    action_mask: Tensor,
) -> Tensor: ...
```

Requirements:

- calculate `log_softmax` in FP32 even if model logits are BF16;
- return a documented dtype;
- reject shape mismatches;
- avoid `-100` label magic inside core functions; collators may use it for compatibility, but masks remain the source of truth;
- ensure a zero-active-token sequence raises a data error rather than producing a silent zero denominator.

## 3. Supervised fine-tuning

### 3.1 Objective

NanoPT uses completion-only negative log likelihood:

$$
\mathcal{L}_{\text{SFT}}
=
-
\frac{
\sum_{i,t} m_{i,t}
\log \pi_\theta(y_{i,t} \mid x_i, y_{i,<t})
}{
\sum_{i,t} m_{i,t}
}.
$$

Prompt tokens, padding, and tokens after termination are excluded.

### 3.2 Reference behavior

- Base model: `Qwen/Qwen3-0.6B-Base`.
- Adaptation: BF16 LoRA, not quantized in the initial reference path.
- Targets: a proposed set of attention and MLP projections, validated at runtime.
- Optimizer: AdamW over trainable adapter parameters only.
- Gradient clipping: global norm, explicitly configured and logged.
- Schedule: warmup plus cosine or linear decay; choose one in an ADR and keep it visible.
- Evaluation: fixed held-out prompts with deterministic and sampled modes.

### 3.3 Data rendering

Render prompt and completion separately enough to construct an exact action mask. The implementation must test that concatenating prompt tokens and completion tokens corresponds to the intended chat-template rendering. Do not infer the prompt boundary by searching decoded strings.

### 3.4 SFT pseudocode

```python
model = load_base_model()
policy = attach_new_lora_adapter(model, name="sft")
optimizer = AdamW(trainable_parameters(policy), ...)

for batch in dataloader:
    logits = policy(batch.input_ids, attention_mask=batch.attention_mask).logits
    token_logps = causal_token_logps(logits, batch.input_ids)
    loss = -masked_mean(token_logps, batch.action_mask[:, 1:])
    loss = loss / gradient_accumulation_steps
    loss.backward()

    if accumulation_boundary:
        clip_grad_norm_(...)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

### 3.5 Required SFT metrics

- train and validation completion NLL;
- completion-token accuracy as a diagnostic;
- gradient norm and clipping count;
- learning rate;
- active tokens per optimizer step;
- tokens per second;
- peak allocated/reserved VRAM;
- held-out parse rate and exact-answer accuracy;
- response length.

## 4. Direct Preference Optimization

### 4.1 Preference record

Each record contains:

```text
prompt x
chosen completion y_w
rejected completion y_l
pair source and rejection type
```

The reference policy is the frozen SFT checkpoint. The trainable DPO policy begins as an exact copy of that SFT policy.

### 4.2 Objective

Define policy and reference preference margins:

$$
\Delta_\theta
=
\log \pi_\theta(y_w \mid x)
-
\log \pi_\theta(y_l \mid x),
$$

$$
\Delta_{\text{ref}}
=
\log \pi_{\text{ref}}(y_w \mid x)
-
\log \pi_{\text{ref}}(y_l \mid x).
$$

The standard DPO loss is:

$$
\mathcal{L}_{\text{DPO}}
=
-
\log \sigma
\left(
\beta
\left[
\Delta_\theta - \Delta_{\text{ref}}
\right]
\right).
$$

The batch loss is the mean over pairs.

### 4.3 Reference-log-probability cache

The default single-GPU path precomputes:

- chosen reference sequence log probability;
- rejected reference sequence log probability;
- active chosen/rejected token counts;
- cache fingerprint metadata.

The cache is invalid if any of these change:

- base model or tokenizer revision;
- SFT adapter content hash;
- renderer version or chat template hash;
- dataset fingerprint;
- max lengths or truncation policy;
- EOS inclusion policy;
- sequence reduction convention.

### 4.4 DPO implementation

```python
for batch in preference_loader:
    chosen_logits = policy(...chosen...).logits
    rejected_logits = policy(...rejected...).logits

    chosen_logps = completion_sequence_logps(...)
    rejected_logps = completion_sequence_logps(...)

    policy_margin = chosen_logps - rejected_logps
    reference_margin = batch.ref_chosen_logps - batch.ref_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    loss = -logsigmoid(logits).mean()
```

For efficiency, chosen and rejected examples may be concatenated into one model forward if the resulting memory behavior is measured and the code remains readable.

### 4.5 Required DPO metrics

- DPO loss;
- policy chosen/rejected log probabilities;
- policy margin;
- reference margin;
- implicit reward margin $\beta(\Delta_\theta-\Delta_{\text{ref}})$;
- preference accuracy, `policy_margin > 0`;
- reward accuracy, `implicit_reward_margin > 0`;
- chosen/rejected active token lengths;
- held-out task metrics and regressions;
- KL/drift diagnostics relative to SFT on a fixed probe set.

### 4.6 Required tests

- zero loss symmetry behavior when policy and reference margins match;
- lower loss when the policy increases the chosen margin;
- correct sign for rejected preference;
- beta scaling;
- masking and padding invariance;
- equality between cached and live reference calculations on a fixture.

## 5. Policy-gradient foundation

For an action $a_t$ sampled in state $s_t$, the policy-gradient direction is:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
A_t
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)
\right].
$$

Intuition:

- if $A_t > 0$, increase the sampled action's probability;
- if $A_t < 0$, decrease it;
- a baseline changes variance without changing the ideal expected gradient when constructed correctly.

NanoPT must explain that a terminal response reward does not identify which token or tool step caused success. Assigning one response-level advantage to every generated token is a practical but coarse credit-assignment choice.

## 6. PPO teaching implementation

PPO is not part of the full 0.6B reference pipeline in v0.1. A tiny implementation must nevertheless expose:

- rollout collection;
- return calculation;
- a learned value baseline;
- generalized advantage estimation;
- old-policy log probabilities;
- clipped policy loss;
- clipped or standard value loss;
- entropy bonus;
- minibatch epochs.

The probability ratio is:

$$
r_t(\theta)
=
\exp
\left(
\log \pi_\theta(a_t \mid s_t)
-
\log \pi_{\text{old}}(a_t \mid s_t)
\right).
$$

The clipped policy objective is:

$$
\mathcal{L}_{\text{PPO-policy}}
=
-
\mathbb{E}_t
\left[
\min
\left(
 r_t A_t,
 \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t
\right)
\right].
$$

The lab must include examples showing why the `min` behaves differently for positive and negative advantages.

## 7. GRPO / RLVR

### 7.1 Rollout grouping

For each prompt $x_i$, sample $G$ completions:

$$
y_{i,1}, \ldots, y_{i,G}
\sim
\pi_{\text{old}}(\cdot \mid x_i).
$$

Each completion is verified to produce scalar reward $r_{i,g}$ and component rewards.

Reference requirement: $G \ge 2$. The initial proposed value is $G=4$, subject to GPU calibration.

### 7.2 Group-relative advantages

Reference mode `group_zscore` uses population standard deviation:

$$
\mu_i
=
\frac{1}{G}
\sum_{g=1}^{G} r_{i,g},
$$

$$
\sigma_i
=
\sqrt{
\frac{1}{G}
\sum_{g=1}^{G}
(r_{i,g}-\mu_i)^2
},
$$

$$
A_{i,g}
=
\frac{r_{i,g}-\mu_i}{\sigma_i+\epsilon_A}.
$$

Use `unbiased=False` in PyTorch. If all rewards are equal, advantages are exactly or numerically zero and the group supplies no policy-gradient signal. Log the fraction of such groups.

Also implement `group_centered` for experiments:

$$
A_{i,g} = r_{i,g} - \mu_i.
$$

Do not label these variants as equivalent. Standard-deviation scaling changes the weighting of groups with different reward dispersion.

### 7.3 Exact rollout policy

The reference training sampler uses:

```yaml
temperature: 1.0
top_p: 1.0
top_k: null
```

The sampler records the log probability of each sampled token under the exact distribution from which it was sampled. Generation occurs under `torch.no_grad()` with the policy adapter in evaluation mode.

### 7.4 Clipped policy loss

For each active generated token:

$$
\rho_{i,g,t}(\theta)
=
\exp
\left(
\log \pi_\theta(y_{i,g,t} \mid x_i, y_{i,g,<t})
-
\log \pi_{\text{old}}(y_{i,g,t} \mid x_i, y_{i,g,<t})
\right).
$$

Broadcast response-level advantage $A_{i,g}$ to active completion tokens. Define:

$$
\ell_{i,g,t}
=
-
\min
\left(
\rho_{i,g,t} A_{i,g},
\operatorname{clip}
(\rho_{i,g,t},1-\epsilon,1+\epsilon)
A_{i,g}
\right).
$$

### 7.5 Loss normalization

Implement and name the normalization explicitly.

`token_mean`:

$$
\mathcal{L}_{\text{policy}}
=
\frac{
\sum_{i,g,t} m_{i,g,t}\ell_{i,g,t}
}{
\sum_{i,g,t} m_{i,g,t}
}.
$$

`sequence_mean`:

$$
\mathcal{L}_{\text{policy}}
=
\frac{1}{N G}
\sum_{i,g}
\frac{
\sum_t m_{i,g,t}\ell_{i,g,t}
}{
\sum_t m_{i,g,t}
}.
$$

The validated reference setting is `token_mean`. The M6 and M7 reports display the chosen mode and
bind it to the exact experiment config; alternative normalizations remain experimental.

### 7.6 Optional KL regularization

Let:

$$
d_{i,g,t}
=
\log \pi_\theta(y_{i,g,t} \mid \cdot)
-
\log \pi_{\text{ref}}(y_{i,g,t} \mid \cdot).
$$

A nonnegative sampled KL estimator may be computed as:

$$
k_{i,g,t}
=
\exp(-d_{i,g,t}) + d_{i,g,t} - 1.
$$

Then:

$$
\mathcal{L}
=
\mathcal{L}_{\text{policy}}
+
\beta_{\text{KL}}
\operatorname{MaskedMean}(k).
$$

The reference recipe may initially use $\beta_{\text{KL}}=0$ to simplify the single-GPU path. This must never be described as “no drift constraint”: PPO clipping and limited update size are local constraints, while the report separately measures drift from the stage-start checkpoint.

### 7.7 Rollout batch and updates

The initial implementation is synchronous:

```text
sample prompt batch
→ generate all G completions per prompt
→ parse and verify
→ calculate group advantages
→ freeze rollout token IDs and old log probabilities
→ perform one or more minibatch epochs
→ discard rollout batch
→ generate fresh data with updated policy
```

Default to one update epoch initially. Additional epochs are an explicit experiment because they increase off-policy distance from the rollout policy.

### 7.8 GRPO pseudocode

```python
for iteration in range(num_iterations):
    prompts = prompt_sampler.next_batch()

    with torch.no_grad():
        rollout = sampler.generate_grouped(
            policy=policy,
            prompts=prompts,
            group_size=G,
        )
        rewards = reward_pipeline(rollout, tasks)
        advantages = group_relative_advantages(rewards, ...)
        if kl_beta > 0:
            reference_logps = score_with_reference_adapter(rollout)

    for minibatch in make_minibatches(rollout, advantages):
        current_logits = policy(minibatch.full_input_ids, ...).logits
        current_logps = selected_action_logps(...)
        ratio = exp(current_logps - minibatch.old_logps)
        policy_loss = clipped_policy_loss(
            ratio,
            minibatch.advantages,
            minibatch.action_mask,
            ...,
        )
        kl_loss = optional_sampled_kl(...)
        loss = policy_loss + kl_beta * kl_loss
        loss.backward()
        optimizer_step_if_ready()
```

### 7.9 Required GRPO metrics

Per iteration and aggregated:

- total and component reward mean/std/min/max;
- pass rate and parser success rate;
- group reward standard deviation;
- degenerate all-equal group fraction;
- advantage mean/std/max absolute value;
- completion length mean and quantiles;
- EOS/maximum-length finish fractions;
- current-minus-old log-probability mean;
- approximate KL to old policy;
- optional KL to reference;
- ratio mean and quantiles;
- clip fraction;
- entropy estimate;
- policy loss and total loss;
- active tokens and prompts per update;
- rollout and training wall-clock fractions;
- generation and training throughput;
- peak VRAM;
- reward-hacking examples.

### 7.10 Required GRPO tests

- group advantages sum approximately to zero per group;
- all-equal rewards produce zero advantages;
- population-std behavior matches a hand calculation;
- ratio is one when current and old log probabilities match;
- clipping behavior for positive and negative advantages;
- padding and post-EOS invariance;
- `token_mean` and `sequence_mean` differ on unequal lengths in the expected way;
- exact sampler log probabilities match a direct forward pass when sampling is untruncated at temperature one;
- saved and reloaded trajectories preserve token IDs and masks exactly.

## 8. Reward and verifier contract

The reference arithmetic reward pipeline returns named components:

```text
parser_valid
format_reward
correctness_reward
optional_length_penalty
verifier_status
```

Proposed initial reward:

$$
r
=
1.0 \cdot r_{\text{correct}}
+
0.1 \cdot r_{\text{format}}
-
\lambda_{\text{length}} p_{\text{length}}.
$$

`length_penalty` is disabled in the first baseline. The strict parser accepts exactly one final answer field in the documented format. Multiple answers, malformed tags, non-finite numbers, trailing answer fields, or parser exceptions do not receive correctness credit.

The raw model text must be retained so reward-hacking behavior can be reviewed.

## 9. Entropy and drift

Full-vocabulary entropy can be expensive but is feasible for sampled action positions in the small reference model. The implementation may log:

$$
H_t
=
-
\sum_v
\pi_\theta(v \mid s_t)
\log \pi_\theta(v \mid s_t)
$$

on a sampled subset of positions. If a cheaper proxy is used, name it accurately.

Checkpoint drift should be measured on a fixed probe set using token-level KL or cross-entropy differences. Do not compare only parameter norms; LoRA parameter distance is not a direct behavioral metric.

## 10. Numerical stability

- compute log-softmax and loss reductions in FP32;
- clamp exponent inputs only when mathematically justified and log clamp events;
- protect denominators with explicit epsilon and validate active counts;
- detect non-finite rewards, advantages, ratios, losses, and gradients;
- store old log probabilities in FP32;
- never exponentiate unbounded differences without diagnostics;
- log maximum absolute log-ratio before clipping.

## 11. Gradient accumulation semantics

Metrics must distinguish:

- microbatch;
- rollout batch;
- minibatch;
- optimizer step;
- effective prompts/completions/tokens per optimizer step.

Divide the loss by accumulation steps exactly once. Clip gradients only at an optimizer boundary. Save and restore accumulation position only if mid-step checkpointing is intentionally supported; otherwise checkpoint only at clean boundaries.

## 12. Algorithm parity tests

After white-box implementations pass, optional tests may compare one small batch with a current TRL implementation. A difference is not automatically a bug because modern trainers expose multiple objective variants. The parity report must align:

- masking;
- sequence reduction;
- reference policy;
- beta/KL setting;
- advantage normalization;
- old-policy handling;
- loss normalization;
- sampling distribution.
