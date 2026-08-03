"""Deterministic, verifier-audited arithmetic preference construction."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import Field

from nanopt.data.arithmetic import (
    canonical_fraction,
    evaluate_ast,
    render_expression,
    render_trusted_completion,
)
from nanopt.data.fingerprints import fingerprint_records
from nanopt.data.schemas import ArithmeticTask, DataModel, SplitName
from nanopt.eval.verifier import verify_task_response
from nanopt.runtime.artifacts import read_jsonl

RejectionType = Literal["wrong_answer", "malformed_answer", "trailing_content"]
PREFERENCE_GENERATOR_VERSION: Literal["controlled-arithmetic-preferences-v1"] = (
    "controlled-arithmetic-preferences-v1"
)


class PreferencePair(DataModel):
    """One chosen/rejected pair with enough lineage to audit its construction."""

    schema_version: Literal[1] = 1
    pair_id: str
    task_id: str
    family: str
    difficulty: int = Field(ge=1, le=5)
    split: SplitName
    prompt: str
    chosen: str
    rejected: str
    rejection_type: RejectionType
    source_dataset_fingerprint: str
    generator_version: Literal["controlled-arithmetic-preferences-v1"] = (
        PREFERENCE_GENERATOR_VERSION
    )
    seed: int


class PreferenceAudit(DataModel):
    """Construction-time audit proving the intended chosen/rejected contract."""

    schema_version: Literal[1] = 1
    pair_count: int = Field(gt=0)
    dataset_fingerprint: str
    source_dataset_fingerprint: str
    rejection_type_counts: dict[str, int]
    split_counts: dict[str, int]
    chosen_character_mean: float = Field(gt=0)
    rejected_character_mean: float = Field(gt=0)
    rejected_to_chosen_character_ratio: float = Field(gt=0)
    all_chosen_correct: Literal[True]
    all_rejected_match_intended_failure: Literal[True]


def _wrong_answer(task: ArithmeticTask) -> str:
    """Construct a canonical but incorrect answer without asking a model."""

    expected = Fraction(evaluate_ast(task.canonical_ast))
    wrong = expected + 1
    _answer_type, wrong_text = canonical_fraction(wrong)
    expression = render_expression(task.canonical_ast)
    return f"<solution>{expression} = {wrong_text}.</solution>\n<answer>{wrong_text}</answer>"


def _rejected_completion(task: ArithmeticTask, rejection_type: RejectionType) -> str:
    chosen = render_trusted_completion(task)
    if rejection_type == "wrong_answer":
        return _wrong_answer(task)
    if rejection_type == "malformed_answer":
        # Keep the derivation and answer text while corrupting only the protocol delimiter. This
        # makes the negative educationally controlled instead of making it nonsense.
        return chosen.replace("<answer>", "<answr>", 1)
    if rejection_type == "trailing_content":
        return chosen + "\nThe answer above is final."
    raise AssertionError(f"unhandled rejection type: {rejection_type}")


def _matches_intended_failure(task: ArithmeticTask, pair: PreferencePair) -> bool:
    result = verify_task_response(task, pair.rejected)
    if pair.rejection_type == "wrong_answer":
        return result.parser.valid and not result.correct
    if pair.rejection_type == "malformed_answer":
        return result.parser.status == "malformed_answer"
    return result.parser.status == "trailing_content"


def generate_preference_pairs(
    tasks: Sequence[ArithmeticTask],
    *,
    source_dataset_fingerprint: str,
    seed: int,
) -> tuple[list[PreferencePair], PreferenceAudit]:
    """Create one reproducible controlled negative for every train/validation task.

    Protected test tasks are rejected deliberately: preference construction must never turn the
    final evaluation split into training or recipe-selection data.
    """

    selected = [task for task in tasks if task.split in {"train", "validation"}]
    if not selected:
        raise ValueError("preference construction requires train or validation tasks")
    if not source_dataset_fingerprint:
        raise ValueError("source dataset fingerprint must not be empty")

    rng = random.Random(seed)
    ordered = sorted(selected, key=lambda task: task.task_id)
    rng.shuffle(ordered)
    rejection_types: tuple[RejectionType, ...] = (
        "wrong_answer",
        "malformed_answer",
        "trailing_content",
    )
    pairs: list[PreferencePair] = []
    tasks_by_id = {task.task_id: task for task in selected}
    for index, task in enumerate(ordered):
        if task.split is None:  # Narrowing; selected tasks necessarily have a split.
            raise AssertionError("selected preference task is missing its split")
        rejection_type = rejection_types[index % len(rejection_types)]
        identity = (
            f"{PREFERENCE_GENERATOR_VERSION}\0{seed}\0{task.task_id}\0{rejection_type}".encode()
        )
        pairs.append(
            PreferencePair(
                pair_id=f"pref_{hashlib.sha256(identity).hexdigest()[:20]}",
                task_id=task.task_id,
                family=task.family,
                difficulty=task.difficulty,
                split=task.split,
                prompt=task.prompt,
                chosen=render_trusted_completion(task),
                rejected=_rejected_completion(task, rejection_type),
                rejection_type=rejection_type,
                source_dataset_fingerprint=source_dataset_fingerprint,
                seed=seed,
            )
        )

    for pair in pairs:
        task = tasks_by_id[pair.task_id]
        if not verify_task_response(task, pair.chosen).correct:
            raise ValueError(f"chosen completion failed verification for {pair.pair_id}")
        if not _matches_intended_failure(task, pair):
            raise ValueError(
                f"rejected completion violated its failure contract for {pair.pair_id}"
            )

    fingerprint = fingerprint_records(
        [pair.model_dump(mode="json") for pair in pairs],
        namespace="arithmetic-preferences-v1",
    )
    chosen_lengths = [len(pair.chosen) for pair in pairs]
    rejected_lengths = [len(pair.rejected) for pair in pairs]
    chosen_mean = sum(chosen_lengths) / len(chosen_lengths)
    rejected_mean = sum(rejected_lengths) / len(rejected_lengths)
    audit = PreferenceAudit(
        pair_count=len(pairs),
        dataset_fingerprint=fingerprint,
        source_dataset_fingerprint=source_dataset_fingerprint,
        rejection_type_counts=dict(sorted(Counter(pair.rejection_type for pair in pairs).items())),
        split_counts=dict(sorted(Counter(pair.split for pair in pairs).items())),
        chosen_character_mean=chosen_mean,
        rejected_character_mean=rejected_mean,
        rejected_to_chosen_character_ratio=rejected_mean / chosen_mean,
        all_chosen_correct=True,
        all_rejected_match_intended_failure=True,
    )
    return pairs, audit


def read_preference_pairs(path: Path) -> list[PreferencePair]:
    """Read strict preference JSONL and reject duplicate pair IDs."""

    pairs = [PreferencePair.model_validate(value, strict=True) for value in read_jsonl(path)]
    if not pairs:
        raise ValueError("preference file must contain at least one pair")
    pair_ids = [pair.pair_id for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("preference file contains duplicate pair IDs")
    return pairs
