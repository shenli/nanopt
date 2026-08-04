# From a tool call to a trajectory

## Learning objectives

After this chapter, you should be able to:

- distinguish environment state, model observation, structured action, and tool result;
- trace one episode through reset, step, termination, and final verification;
- explain why invalid actions still consume budgets;
- inspect a retained trajectory without trusting its aggregate score;
- state why this v0.1 environment evaluates policies but does not train them.

## The environment is a state machine

A coding agent is not just a language model with a long prompt. It is a stateful loop whose
transitions must be specified and audited:

```text
immutable snapshot
        │ reset + hash check
        ▼
visible workspace ──► observation ──► one JSON action
        ▲                                   │
        │                                   ▼
        └──────── trusted allow-listed tool + budget charge
                                                │
                                  finish, timeout, or exhaustion
                                                ▼
                              separate public + hidden verification
```

[`MiniSWEEnvironment`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/environment.py)
owns this loop. A reset copies only the task's `snapshot/` directory into a fresh temporary
workspace and checks its SHA-256 against the task card. The model receives the issue, available
tools, remaining budgets, prior action/result transcript, and latest result. It never receives the
trusted hidden-test directory.

The policy returns exactly one JSON object. For example:

```json
{
  "tool": "read_file",
  "arguments": {"path": "src/range_utils.py", "start_line": 1, "end_line": 200}
}
```

[`parse_action`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/records.py)
validates that object against a strict discriminated union. Unknown fields, unknown tools, prose
around the JSON, and shell commands are invalid. This is a protocol boundary, not merely a prompt
request.

## Why invalid actions cost a turn

An agent could otherwise probe the parser or security boundary for free until one attempt works.
NanoPT charges the turn and tool-call budget before parsing and executing the action. The invalid
response, structured error, budgets before and after, and policy-violation code all remain in the
trajectory. Failures therefore stay visible and have a bounded cost.

The environment terminates when the policy calls `finish`, exhausts its turn or tool budget,
reaches the wall-clock deadline, or causes a policy/sandbox failure. Termination does not determine
the score. Final public and hidden verification always does.

## What a trajectory proves

Each `AgentTrajectory` records:

- task, environment, policy, and checkpoint identity;
- immutable initial snapshot hash;
- the observation shown at each turn;
- raw model response and exact sampled token IDs when a model produced it;
- parsed action, bounded result, budget transition, and workspace hash;
- termination reason and separate public/hidden summaries;
- final score and policy-violation penalty.

The scripted oracle is intentionally unprivileged: it submits its reviewed patch through the same
`apply_patch` action, runs the same public-test tool, and requests the same final verification as a
model. Replaying its retained response objects from a fresh reset must produce the same semantic
trajectory. Timing and captured test text are normalized because they are measurements, not state
transition semantics.

## Exact tokens are retained, but no optimization occurs

The Qwen policy adapter renders the observation, samples one response, and records the exact
generated token IDs before decoding the JSON view. This preserves the action identity needed by
Agent SFT or Agent RL. The environment-only validation never computes an agent loss,
backpropagates, or updates a checkpoint. Its reports say **evaluation only** for that reason.

Run the CPU lab:

```bash
uv run python labs/10_mini_swe_environment.py
```

Then inspect the concrete contracts in this order:

1. [`records.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/records.py)
2. [`workspace.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/workspace.py)
3. [`environment.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/environment.py)
4. [`verifier.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/verifier.py)
5. [`run.py`](https://github.com/shenli/nanopt/blob/main/src/nanopt/agent/run.py)

## Common mistakes

- Treating model text as a shell command.
- Resetting from the previous episode instead of an immutable snapshot.
- Omitting invalid attempts from budgets or logs.
- Equating `finish` with success before verification.
- Recording only the final score and losing the action/state evidence.
- Calling stored token IDs “Agent RL” when no policy update exists.
