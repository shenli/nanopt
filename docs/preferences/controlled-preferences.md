# Controlled preference data

## Learning objectives

After this chapter, you should be able to:

- explain why a chosen/rejected pair needs a specific, auditable reason;
- distinguish an incorrect answer from a parser failure;
- identify length and rejection-type shortcuts before training;
- reproduce NanoPT's preference dataset and fingerprint;
- explain why protected evaluation tasks never enter preference construction.

## A pair is a claim, not just two strings

A preference record says that completion $y_w$ should be preferred to $y_l$ for prompt $x$. If the
record does not say *why*, a learner cannot tell whether training rewards correctness, formatting,
verbosity, or an accidental artifact.

NanoPT constructs the chosen completion directly from the trusted arithmetic AST. It constructs one
of three controlled rejected completions:

| Rejection type | What changes | Required verifier outcome |
| --- | --- | --- |
| `wrong_answer` | The exact value changes by one | parses, but is incorrect |
| `malformed_answer` | The opening answer tag is misspelled | malformed-answer failure |
| `trailing_content` | Non-whitespace follows the answer | trailing-content failure |

[`generate_preference_pairs`](https://github.com/shenli/nanopt/blob/main/src/nanopt/data/preferences.py)
independently verifies every chosen and rejected completion. Construction stops if a chosen answer
fails or a rejected answer does not produce its declared failure.

## Leakage boundary

Only `train` and `validation` tasks can become preference pairs. The protected IID,
compositional, range, format-attack, and smoke evaluation splits are excluded in code. This keeps
the final generation evaluator useful: DPO cannot memorize the exact protected prompts through its
offline pair dataset.

## Fingerprints and audit

The preference fingerprint covers the full ordered records, including:

- source task and pair IDs;
- chosen and rejected text;
- rejection type and split;
- generator version and seed;
- source arithmetic-dataset fingerprint.

`preference_audit.json` records pair counts, rejection and split distributions, verifier-contract
results, and chosen/rejected character lengths. Token lengths are measured again after rendering in
the DPO run because tokenizer boundaries—not characters—control the objective.

Length balance is an audit signal, not a promise that every candidate has identical length. A large
system might match or stratify lengths. NanoPT instead keeps three transformations easy to inspect
and requires the report to disclose their length distributions and per-type behavior.

## Run the CPU lab

```bash
uv run python labs/08_controlled_preferences.py
```

To create the complete local artifact after generating arithmetic tasks:

```bash
uv run nanopt data generate
uv run nanopt data preferences
```

Inspect `artifacts/data/arithmetic_preferences_v1/preference_audit.json` before training.

## Common mistakes

- Using protected prompts to create negatives or tune rejection mixtures.
- Assuming any rejected string is useful merely because it differs from the chosen string.
- Mixing parser failures and incorrect answers into one opaque reward label.
- Letting one rejection type or length cue dominate without reporting it.
- Changing pair text without changing the dataset fingerprint.

At production scale, preference collection also needs annotator policy, disagreement handling,
privacy review, deduplication, source licensing, and distribution monitoring. The controlled NanoPT
generator isolates the mechanics; it is not a substitute for those systems.
