from __future__ import annotations

from nanopt.data.arithmetic import generate_task
from nanopt.data.preferences import generate_preference_pairs


def _tasks() -> list:
    tasks = []
    for index in range(9):
        task = generate_task(family="addition_subtraction", difficulty=1, seed=index)
        split = "train" if index < 6 else ("validation" if index < 8 else "test_iid")
        tasks.append(task.model_copy(update={"split": split}))
    return tasks


def test_controlled_preferences_are_reproducible_and_exclude_protected_tasks() -> None:
    pairs, audit = generate_preference_pairs(
        _tasks(), source_dataset_fingerprint="source-v1", seed=17
    )
    repeated, repeated_audit = generate_preference_pairs(
        _tasks(), source_dataset_fingerprint="source-v1", seed=17
    )

    assert pairs == repeated
    assert audit == repeated_audit
    assert len(pairs) == 8
    assert {pair.split for pair in pairs} == {"train", "validation"}
    assert audit.all_chosen_correct is True
    assert audit.all_rejected_match_intended_failure is True
    assert audit.rejection_type_counts == {
        "malformed_answer": 3,
        "trailing_content": 2,
        "wrong_answer": 3,
    }


def test_preference_fingerprint_changes_with_seeded_rejection_assignment() -> None:
    _pairs_a, audit_a = generate_preference_pairs(
        _tasks(), source_dataset_fingerprint="source-v1", seed=1
    )
    _pairs_b, audit_b = generate_preference_pairs(
        _tasks(), source_dataset_fingerprint="source-v1", seed=2
    )

    assert audit_a.dataset_fingerprint != audit_b.dataset_fingerprint
