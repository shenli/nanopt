"""Filter synthetic delayed outcomes into a privacy-minimized task queue."""

from __future__ import annotations

from nanopt.systems.flywheel import SessionSignal, build_task_candidates


def main() -> None:
    """Keep failures with consent while excluding sensitive and fixed-evaluation signals."""

    common = {
        "task_family": "configuration",
        "failure_code": "boolean-case",
        "outcome_success": False,
        "outcome_delay_hours": 24,
        "research_consent": True,
        "contains_sensitive_data": False,
    }
    signals = [
        SessionSignal(session_id="synthetic-a", **common),
        SessionSignal(session_id="synthetic-b", **common),
        SessionSignal(session_id="private", **(common | {"research_consent": False})),
        SessionSignal(session_id="sensitive", **(common | {"contains_sensitive_data": True})),
        SessionSignal(session_id="protected", **(common | {"in_fixed_evaluation": True})),
    ]
    candidates, audit = build_task_candidates(signals)

    print(f"Input signals:       {audit.input_signals}")
    print(f"Accepted failures:   {audit.accepted_failure_signals}")
    print(f"Candidate patterns:  {len(candidates)}")
    print(f"Source digest:       {candidates[0].source_ids_sha256}")
    assert candidates[0].observations == 2
    assert audit.rejected_no_consent == 1
    assert audit.rejected_sensitive == 1
    assert audit.rejected_fixed_evaluation == 1
    print("Production-flywheel simulation passed.")


if __name__ == "__main__":
    main()
