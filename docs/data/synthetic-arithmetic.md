# Deterministic synthetic arithmetic data

## Learning objectives

After this lesson, you should be able to:

- explain why NanoPT generates structured tasks before rendering text;
- evaluate an arithmetic AST without Python `eval`;
- distinguish task IDs, canonical task hashes, and dataset fingerprints;
- construct splits without leaking equivalent tasks across boundaries;
- separate answer parsing from correctness verification;
- reproduce a small dataset fingerprint from one checked-in configuration.

## Generate structure before prose

NanoPT's reference domain is deliberately narrow arithmetic. The objective is to make post-training
mechanics measurable, not to claim broad mathematical intelligence.

Each task begins as a recursive AST containing integer literals and one of four operations: add,
subtract, multiply, or divide. [`evaluate_ast`](https://github.com/shenli/nanopt/blob/main/src/nanopt/data/arithmetic.py)
uses Python's exact `Fraction` arithmetic. It never calls `eval`, and trusted answers never come from
a language model.

The initial generator provides four controlled families:

- addition and subtraction;
- multiplication;
- exact integer division;
- mixed, fully parenthesized operations.

Only after the AST and answer exist does the generator render a prompt. A trusted completion is also
rendered directly from the AST and exact answer.

## Three different identifiers

**Task ID** identifies one canonical generated task record. It is derived from family, difficulty,
and AST.

**Canonical task hash** covers task structure and operands before prompt rendering. Split leakage
checks use this hash, so changing a paraphrase cannot move an equivalent task into another split.

**Dataset fingerprint** covers:

- the complete generator configuration and master seed;
- parser and verifier versions;
- every complete task record in stable task-ID order.

Changing any of those inputs changes the fingerprint.

## Leakage-safe splits

[`build_splits`](https://github.com/shenli/nanopt/blob/main/src/nanopt/data/splits.py) first rejects
duplicate canonical hashes. It then sorts tasks by a hash of the split seed and canonical task hash
and slices that order into the seven named splits. Every task must be assigned exactly once.

The manifest records per-split counts and canonical hashes. Tests flatten those hashes and prove that
their union contains no duplicates.

## Parser and verifier are separate

The strict parser accepts exactly one lowercase final `<answer>...</answer>` field. It rejects
multiple answers, malformed or case-shifted tags, tag attributes, trailing content, excessive
length, markup inside the value, floats where integers are expected, NaN, infinity, and ambiguous
integer spellings.

Parsing answers only: "is there one well-formed candidate?" The verifier independently evaluates the
trusted AST, checks that the stored target has not been corrupted, and then compares the parsed
candidate. Reports can therefore distinguish format failure from a well-formed wrong answer.

## CPU lab

Run the complete in-memory path:

```bash
uv run python labs/05_synthetic_arithmetic.py
```

The lab loads the checked-in generator configuration, generates the same tasks twice, compares
fingerprints, constructs all splits, renders one trusted answer, and verifies it exactly.

## Limitations

This layer establishes deterministic mechanics, not a claim about broad dataset scale or
curriculum quality. Later additions may introduce controlled task families and preference
transformations, but every addition must bring property tests, canonical leakage checks, parser
attacks, and updated fingerprints.

The dataset card and starting configuration live in
[`tasks/arithmetic`](https://github.com/shenli/nanopt/tree/main/tasks/arithmetic).
