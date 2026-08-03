# Data and Tasks

## 1. Golden-path data strategy

The official v0.1 pipeline uses generated, exact-answer tasks rather than downloading a large external instruction dataset. This keeps licensing, leakage, reproducibility, verification, and runtime under project control.

The reference domain is arithmetic and small symbolic reasoning. It is deliberately narrow: the goal is to make post-training mechanics measurable, not to claim general assistant quality.

## 2. Task representation

A task is generated from a structured abstract syntax tree rather than from an answer string. The canonical task object contains:

```json
{
  "schema_version": 1,
  "task_id": "arith_train_000001",
  "family": "mixed_precedence",
  "difficulty": 3,
  "prompt": "Compute ...",
  "canonical_ast": {"op": "..."},
  "answer_type": "integer",
  "canonical_answer": "42",
  "metadata": {
    "generator_version": "...",
    "seed": 42,
    "split": "train"
  }
}
```

The AST is evaluated with safe project code. Never call Python `eval` on generated or model-produced text.

## 3. Task families

Start with a controlled set whose difficulty can be varied independently.

### 3.1 Integer arithmetic

- addition and subtraction with signed integers;
- multiplication;
- exact integer division;
- multi-operation expressions;
- precedence and parentheses.

### 3.2 Small equations

- one-step linear equations;
- two-step linear equations;
- integer solutions only in the initial version.

### 3.3 Sequence and transformation tasks

- short arithmetic progressions;
- apply a stated series of transformations;
- simple table lookups plus arithmetic.

### 3.4 Structured-output tasks

The same underlying problem is presented with an explicit answer protocol. These tasks separate reasoning correctness from instruction and format following.

Do not add task families until the generator has property tests, leakage-safe canonicalization, and a verifier.

## 4. Output protocol

The initial response format is intentionally simple:

```text
<solution>
A concise derivation.
</solution>
<answer>42</answer>
```

The strict parser requires exactly one `<answer>...</answer>` field. The value must match the expected answer type and canonicalization rules.

The project should avoid relying on XML validity in general; this is a narrow protocol designed for transparent parsing. Parser behavior is versioned and included in data and run fingerprints.

## 5. Record types

### 5.1 SFT record

```json
{
  "schema_version": 1,
  "record_id": "sft_...",
  "task_id": "...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "target": {
    "answer": "42",
    "answer_type": "integer"
  },
  "provenance": {...}
}
```

The assistant response is generated from the trusted AST through deterministic templates. Multiple solution styles may be sampled, but their generator and seed must be recorded.

### 5.2 Preference record

```json
{
  "schema_version": 1,
  "record_id": "pref_...",
  "task_id": "...",
  "prompt_messages": [...],
  "chosen": "...",
  "rejected": "...",
  "rejection_type": "arithmetic_error",
  "target": {...},
  "provenance": {...}
}
```

### 5.3 RL prompt record

```json
{
  "schema_version": 1,
  "record_id": "rl_...",
  "task_id": "...",
  "prompt_messages": [...],
  "target": {...},
  "verifier": {
    "type": "exact_answer",
    "version": "1"
  },
  "provenance": {...}
}
```

### 5.4 Evaluation result

The evaluation result stores the immutable input/task reference, generation configuration, response text, parser outcome, verified answer, reward components, exact token IDs when available, timing, and checkpoint ID.

## 6. Preference-pair construction

Rejected answers must be generated through controlled transformations, not merely by sampling a weak model and assuming its response is wrong.

Required rejection types:

- `arithmetic_error`: one intermediate operation or final answer is wrong;
- `sign_error`;
- `operator_error`;
- `format_error`: answer is correct but protocol is invalid;
- `multiple_answers`: conflicting final answer fields;
- `incomplete_solution`;
- `irrelevant_verbose_response`;
- `correct_reasoning_wrong_final`;
- `wrong_reasoning_accidentally_correct_final`, used cautiously and labeled.

Balance rejection types so DPO cannot solve the dataset using one shortcut such as response length. Log chosen/rejected length distributions and classifier-style heuristics before training.

A preference record must be validated by the same parser and verifier used for evaluation:

- chosen must pass;
- rejected must fail the intended criterion;
- no pair may contain identical rendered completions;
- task answer and chosen answer must agree;
- rejection type must match observed verifier behavior.

## 7. Split construction and leakage control

Split by canonical task structure and operands before rendering text. Never randomly split already rendered paraphrases of the same AST.

Required splits:

- `train`: optimization data;
- `validation`: hyperparameter and early-stopping diagnostics;
- `test_iid`: held-out tasks from the same generator distribution;
- `test_compositional`: held-out operation combinations or equation templates;
- `test_range`: larger or shifted operand ranges;
- `test_format_attack`: prompts and outputs designed to stress parsing and reward hacking;
- `smoke`: tiny deterministic subset for the local validation gate and debugging.

Use canonical hashes to assert no overlap across protected splits. Store a split manifest with counts per family and difficulty.

## 8. Validated teaching scale and larger experiments

The validated M7 teaching pipeline uses one deterministic 128-task corpus across training,
validation, and protected splits. The larger values below remain scaling targets, not support
claims:

| Dataset | Smoke | Proposed reference |
|---|---:|---:|
| SFT train | 128 | 8,000–12,000 |
| SFT validation | 64 | 1,000 |
| Preference train | 128 | 6,000–10,000 |
| Preference validation | 64 | 1,000 |
| RL prompt pool | 64 | 2,000–5,000 |
| Each primary test split | 64 | 500–1,000 |

Keep the generator capable of producing larger datasets without changing schemas. A future larger
reference recipe needs its own frozen targets and evidence; it does not inherit M7 validation.

## 9. Dataset fingerprints

A dataset fingerprint must include:

- schema version;
- generator source version or git commit;
- generator configuration;
- random seed;
- task-family definitions;
- split algorithm version;
- parser/verifier version;
- record content hash.

Reference-log-probability caches and evaluation comparisons must reject a fingerprint mismatch.

## 10. Data quality checks

`nanopt data validate` must report:

- schema-valid record counts;
- duplicate record and task hashes;
- split overlap;
- answer-type distribution;
- task-family and difficulty distribution;
- prompt/completion token-length distributions for the selected tokenizer;
- chosen/rejected length and verifier distributions;
- invalid or ambiguous parses;
- non-finite or out-of-range answers;
- examples from every rejection type;
- exact reproducibility from a sample of generator seeds.

## 11. Curriculum and prompt sampling

v0.1 uses a fixed prompt-pool distribution. Adaptive curriculum is postponed until metrics are stable. A future sampler may upweight tasks whose group rewards contain useful variation, but it must avoid training only on tasks near the current decision boundary and forgetting anchor tasks.

## 12. External datasets

External datasets such as GSM8K may be used in optional comparison labs only after:

- checking license and redistribution terms;
- documenting train/test contamination concerns;
- adapting parsing without weakening the reference verifier;
- keeping them outside the required offline golden path.

## 13. Data publication

Generated datasets may be released separately with a dataset card. The card must explain that they are synthetic educational tasks, list all generator versions, and avoid overstating their usefulness for general reasoning.
