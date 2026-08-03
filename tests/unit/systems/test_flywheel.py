from __future__ import annotations

from nanopt.systems.flywheel import SessionSignal, build_task_candidates


def _signal(session_id: str, **changes: object) -> SessionSignal:
    values: dict[str, object] = {
        "session_id": session_id,
        "task_family": "configuration",
        "failure_code": "boolean-case",
        "outcome_success": False,
        "outcome_delay_hours": 24,
        "research_consent": True,
        "contains_sensitive_data": False,
    }
    values.update(changes)
    return SessionSignal(**values)  # type: ignore[arg-type]


def test_flywheel_filters_privacy_and_fixed_evaluation_boundaries() -> None:
    candidates, audit = build_task_candidates(
        [
            _signal("accepted-a"),
            _signal("accepted-b"),
            _signal("private", research_consent=False),
            _signal("sensitive", contains_sensitive_data=True),
            _signal("fresh", outcome_delay_hours=0),
            _signal("success", outcome_success=True),
            _signal("protected", in_fixed_evaluation=True),
        ]
    )

    assert len(candidates) == 1
    assert candidates[0].observations == 2
    assert "accepted-a" not in candidates[0].source_ids_sha256
    assert audit.accepted_failure_signals == 2
    assert audit.rejected_no_consent == 1
    assert audit.rejected_sensitive == 1
    assert audit.rejected_unresolved == 1
    assert audit.rejected_success == 1
    assert audit.rejected_fixed_evaluation == 1
