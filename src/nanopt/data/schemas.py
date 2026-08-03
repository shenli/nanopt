"""Versioned typed records for generated arithmetic tasks and split manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ArithmeticOperator = Literal["add", "subtract", "multiply", "divide"]
AnswerType = Literal["integer", "rational", "string"]
SplitName = Literal[
    "train",
    "validation",
    "test_iid",
    "test_compositional",
    "test_range",
    "test_format_attack",
    "smoke",
]


class DataModel(BaseModel):
    """Strict base so generated records cannot acquire undocumented fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ArithmeticAst(DataModel):
    """One recursive integer literal or binary arithmetic expression.

    ``kind='integer'`` requires ``value`` and forbids children. ``kind='binary'`` requires ``op``,
    ``left``, and ``right`` and forbids ``value``. A single recursive model keeps JSON serialization
    simple while the validator preserves the two disjoint variants.
    """

    kind: Literal["integer", "binary"]
    value: int | None = None
    op: ArithmeticOperator | None = None
    left: ArithmeticAst | None = None
    right: ArithmeticAst | None = None

    @model_validator(mode="after")
    def validate_variant(self) -> ArithmeticAst:
        if self.kind == "integer":
            if self.value is None or any(
                item is not None for item in (self.op, self.left, self.right)
            ):
                raise ValueError("integer AST nodes require only value")
        elif self.value is not None or self.op is None or self.left is None or self.right is None:
            raise ValueError("binary AST nodes require only op, left, and right")
        return self


class TaskTarget(DataModel):
    answer_type: AnswerType
    canonical_answer: str


class TaskProvenance(DataModel):
    generator_version: str
    seed: int
    source_task_id: str | None = None
    license: str | None = "Apache-2.0"


class VerifierSpec(DataModel):
    type: Literal["exact_answer"] = "exact_answer"
    version: Literal["1"] = "1"


class ArithmeticTask(DataModel):
    """Canonical generated task compatible with ``task.schema.json``."""

    schema_version: Literal[1] = 1
    task_id: str
    task_version: Literal["1"] = "1"
    family: str
    difficulty: int = Field(ge=1, le=5)
    prompt: str
    canonical_ast: ArithmeticAst
    target: TaskTarget
    split: SplitName | None = None
    verifier: VerifierSpec = Field(default_factory=VerifierSpec)
    provenance: TaskProvenance


class SplitManifest(DataModel):
    """Counts and canonical hashes proving deterministic split separation."""

    schema_version: Literal[1] = 1
    split_algorithm_version: Literal["canonical-hash-v1"] = "canonical-hash-v1"
    seed: int
    dataset_fingerprint: str
    counts: dict[SplitName, int]
    canonical_hashes: dict[SplitName, list[str]]
