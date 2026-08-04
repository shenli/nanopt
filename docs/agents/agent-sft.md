# Agent SFT: from replayable trajectories to action targets

Agent SFT is ordinary causal language-model training with an unusually important data contract.
The prompt is not one static question: it is the complete state available before an action. A
mistake in where that state ends changes which tokens receive gradients.

## Learning objectives

After this chapter, you should be able to:

- explain why previous assistant actions are context rather than current targets;
- trace one example back to a replay-checked environment trajectory;
- compare a full alternating transcript with a serialized observation snapshot;
- distinguish action-protocol learning from held-out task-solving ability.

## The exact boundary

For turn $t$, NanoPT renders messages through the assistant generation marker, then renders the
same messages with the trusted action appended. The prompt tokens must be an exact prefix of the
full sequence. In full-sequence coordinates,

$$
m_i = \begin{cases}
1 & \text{if token } i \text{ belongs to the current action,}\\
0 & \text{for system, observations, previous actions, and padding.}
\end{cases}
$$

The causal loss shifts that mask exactly once. Previous tool calls remain visible to the model, but
they are not trained again at every later turn. Run the small proof without downloading a model:

```bash
uv run python labs/20_agent_sft_masks.py
```

The implementation is split intentionally:

1. [`context.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/context.py) defines
   the online/offline message contract.
2. [`sft_data.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/sft_data.py) collects,
   replays, renders, hashes, and freezes examples.
3. [`sft_records.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/sft_records.py)
   validates token and mask coordinates.
4. [`sft_run.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/sft_run.py) consumes
   stored IDs directly and trains only LoRA parameters.

## Demonstrations and recovery

Each reviewed demonstration uses the same environment tools as a model:

```text
list_files → read_file → apply_patch → run_tests → finish
```

Recovery trajectories first submit malformed JSON. The environment charges the turn, records an
`invalid_action` result, and the trusted next action demonstrates how to continue. NanoPT never
trains on the malformed action itself.

Every source episode is replayed in a fresh reset. Dataset examples retain the trajectory ID,
trajectory-file hash, step index, typed target action, rendered messages, token IDs, attention
mask, action mask, prompt length, context policy, and chat-template hash. Loading the dataset checks
those links again.

## Build, train, evaluate

```bash
uv run nanopt agent build-sft-data \
  --output artifacts/data/mini_swe_agent_sft_v1 \
  --local-files-only

uv run nanopt train agent-sft \
  --dataset artifacts/data/mini_swe_agent_sft_v1 \
  --local-files-only \
  --device cuda

uv run nanopt agent run \
  --policy model \
  --experiment agent_sft_eval \
  --task-split all \
  --task-id clamp_reversed_bounds \
  --context-policy full_transcript \
  --adapter artifacts/runs/AGENT_SFT_RUN/adapter/agent_sft \
  --adapter-name agent_sft \
  --local-files-only \
  --device cuda
```

The data command uses the trusted fake backend because it collects reviewed local demonstrations;
behavioral evaluation uses Docker. Hidden-test source is never copied into an observation or
dataset.

## Two context policies

`full_transcript` is a true alternating chat: observation, assistant action, next observation. Each
new observation omits its redundant transcript field. `observation_snapshot` sends one current
observation whose JSON contains the accumulated action/result transcript.

Both expose the same environment facts, but they are not interchangeable distributions. The v0.2
reference comparison trains on `full_transcript`, then evaluates both policies. This is also a
systems experiment: longer prompts increase the cost of NanoPT's intentionally simple reference
sampler.

## Read the metrics honestly

Teacher-forced action NLL answers, “How well does the adapter imitate trusted next tokens?” Action
validity answers, “Can the generated text pass the strict typed protocol?” Hidden-verifier score
answers, “Did the resulting repository actually solve the task?”

A model can improve the first two and still fail a task it never saw. NanoPT reports that outcome
rather than presenting protocol learning as software-engineering generalization.
