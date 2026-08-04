# Agent Environment

## 1. Purpose

The v0.1 agent component teaches why agent post-training requires more than prompt/completion pairs. It provides a small but complete environment with state, observations, tools, transitions, budgets, resets, terminal conditions, and hidden verification.

v0.1 evaluates rollouts only. It does not optimize the language model in this environment. This separation allows the environment contract and security boundary to become stable before adding Agent RL.

## 2. MiniSWE task design

Each task is a small original repository containing a defect or incomplete feature. Tasks should be created specifically for NanoPT rather than copied from third-party repositories.

A task contains:

```text
task ID and version
natural-language issue
initial workspace snapshot
allowed tool set
public test command
hidden verifier specification
budgets
expected terminal artifacts
license metadata
```

Proposed v0.1 scale:

- 5 smoke tasks;
- 20–30 reference tasks;
- Python-only initially;
- repositories small enough to inspect in a few tool calls;
- typical successful trajectories of 3–10 tool calls;
- deterministic tests with no network.

Task categories:

- incorrect edge-case handling;
- parser or serializer bug;
- missing validation;
- small algorithmic defect;
- incomplete function;
- cross-file interface mismatch;
- test-preserving refactor with a hidden requirement.

## 3. Agent protocol

The model receives a system instruction, task description, tool schemas, budget status, and observations. It responds with one structured tool call or `finish` action per turn.

Use a typed JSON action protocol rather than parsing shell commands from free-form text.

Example:

```json
{
  "tool": "read_file",
  "arguments": {
    "path": "src/parser.py",
    "start_line": 1,
    "end_line": 200
  }
}
```

Invalid actions consume a configurable budget unit and return a structured error. They must not be executed partially.

## 4. Required tools

### `list_files`

- lists files beneath a workspace-relative path;
- enforces depth and result limits;
- never follows symlinks outside the workspace.

### `read_file`

- accepts a workspace-relative path and line range;
- validates UTF-8 text or returns a binary-file error;
- applies a maximum byte/line limit.

### `search`

- searches text beneath an allowed path;
- supports a literal query and optional safe glob;
- returns bounded, line-numbered results;
- does not expose hidden-test directories.

### `apply_patch`

- accepts a unified diff;
- validates every target path;
- rejects binary patches, path traversal, symlink escapes, and files outside the workspace;
- applies atomically or returns no modification.

### `run_tests`

- executes a task-defined public test command, not a model-supplied shell command;
- returns bounded stdout/stderr, exit code, duration, and test summary;
- enforces timeout and process limits.

### `finish`

- records the model's completion summary;
- triggers final public-state capture and hidden verification;
- ends the episode.

Do not expose arbitrary `shell`, `python`, package installation, network, Docker, or host filesystem tools in v0.1.

## 5. State and observation

The environment state includes the full sandbox filesystem, consumed budgets, tool-call history, test history, and terminal status. The model observation is a controlled projection of state:

- task statement;
- prior model/tool messages within context policy;
- result of the most recent tool call;
- remaining budgets;
- no hidden tests, verifier source, expected patch, or trusted answer.

This distinction must be explained in the course: the environment may know more than the agent observes.

## 6. Sandbox architecture

### 6.1 Production-like local backend

Use Docker for the reference backend with:

- no network (`--network none`);
- non-root user;
- no Docker socket;
- no GPU device;
- dropped Linux capabilities;
- `no-new-privileges`;
- read-only root filesystem where practical;
- a single read-write workspace mount;
- tmpfs for temporary files;
- memory, CPU, PID, output, and wall-clock limits;
- a pinned image digest in validated runs.

The task workspace must not contain hidden tests.

### 6.2 Unit-test backend

Provide a local fake/in-process backend for protocol and state-machine tests. It must not be described as secure isolation and must never be the default for untrusted model-generated patches.

## 7. Reset and snapshots

The canonical initial task snapshot is immutable. Every episode creates a fresh working copy. A reset must return the same content hash.

Possible v0.1 implementation:

```text
read-only task template
→ copy/reflink into unique temporary workspace
→ run episode
→ hash final tracked files
→ destroy workspace after artifacts are saved
```

Do not rely on `git reset` inside a workspace that the agent can mutate unless the `.git` data is protected outside that workspace.

## 8. Public and hidden verification

Public tests provide diagnostic feedback during the episode. Hidden verification runs after termination in a separate fresh verifier container/workspace:

1. copy only allowed final project files from the episode workspace;
2. inject hidden tests from a protected task bundle;
3. execute the verifier with independent limits;
4. record structured results without exposing hidden source;
5. discard verifier workspace.

The model-facing environment and hidden verifier must not share a writable filesystem.

## 9. Reward and score schema

v0.1 introduced this evaluation score; v0.3 uses the same bounded final score as a hidden outcome
reward after the episode terminates:

```json
{
  "build_ok": true,
  "public_tests_passed": 5,
  "public_tests_total": 5,
  "hidden_tests_passed": 7,
  "hidden_tests_total": 8,
  "policy_violations": 0,
  "budget_exhausted": false,
  "final_score": 0.875
}
```

The implemented score is:

```text
0 if public verification fails
otherwise hidden-tests-passed / hidden-tests-total
minus 0.1 per policy violation, bounded below by 0
```

Never accept the agent's self-reported success as evidence.

## 10. Anti-hacking requirements

The verifier must detect or prevent:

- modifications to test files or task metadata;
- hard-coded expected outputs tied only to public tests;
- faked test-run output;
- attempts to read hidden paths;
- symlink and path-traversal escapes;
- disabling validation or swallowing all exceptions;
- deleting the project and replacing it with a trivial stub when the interface requires more;
- resource-exhaustion attacks;
- nondeterministic timing tricks.

Some behaviors require task-specific hidden tests rather than generic rules. Record new hacking patterns as regression tasks.

## 11. Trajectory schema

Each trajectory records:

- task and environment versions;
- model/checkpoint and generation configuration;
- initial snapshot hash;
- ordered model messages;
- parsed structured actions;
- tool results and errors;
- tool and model timing;
- budget state before/after each action;
- optional workspace state hash at checkpoints;
- finish reason;
- public and hidden verifier summaries;
- final score;
- policy-violation events.

Do not include hidden-test source or secrets. Exact model token IDs should be added in v0.2 when trajectories become training data.

## 12. Agent loop

```python
env = MiniSWEEnvironment(task, sandbox_backend)
observation = env.reset()

while not env.terminated:
    response = policy.generate(observation, tools=env.tool_schemas)
    action = parse_action(response)
    observation = env.step(action)

result = env.finalize_and_verify()
trajectory_writer.write(env.trajectory, result)
```

The environment controls termination on `finish`, budget exhaustion, unrecoverable sandbox failure, or hard timeout.

## 13. Context management

v0.1 uses a bounded full transcript for short trajectories. Do not add summarization or memory systems before exact trajectory semantics are stable. The course should explain that long-horizon systems need context compaction, external memory, cache retention, and resumable state, but those are later milestones.

## 14. Agent RL path

v0.2:

- exact token IDs and action masks;
- tool-call SFT data;
- Agent SFT and behavior cloning;
- outcome and process metrics.

v0.3 implements:

- grouped rollouts from the same task snapshot;
- hidden outcome verifier as reward;
- short-horizon GRPO or PPO-style updates;
- fresh-only updates plus retained fresh-versus-stale policy-drift measurements;
- credit-assignment and tool-budget studies.

The implementation builds on v0.2's deterministic reset, independent hidden verification, and
reproducible trajectory replay. v0.3 deliberately stops at short horizons. The v0.4 systems lab
adds a deterministic model/world checkpoint and partial-rollout control simulation; asynchronous
workers, real KV-cache movement, and accelerated inference remain unimplemented.
