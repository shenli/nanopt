"""Privacy-minimizing synthetic task discovery for a teaching data flywheel."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionSignal:
    """A delayed synthetic outcome signal; raw session content is intentionally absent."""

    session_id: str
    task_family: str
    failure_code: str
    outcome_success: bool
    outcome_delay_hours: int
    research_consent: bool
    contains_sensitive_data: bool
    in_fixed_evaluation: bool = False


@dataclass(frozen=True)
class TaskCandidate:
    """Aggregated failure pattern safe for a human task-authoring queue."""

    task_family: str
    failure_code: str
    observations: int
    source_ids_sha256: str


@dataclass(frozen=True)
class FlywheelAudit:
    """Filtering counts make privacy and evaluation boundaries inspectable."""

    input_signals: int
    accepted_failure_signals: int
    rejected_no_consent: int
    rejected_sensitive: int
    rejected_unresolved: int
    rejected_success: int
    rejected_fixed_evaluation: int


def build_task_candidates(
    signals: list[SessionSignal], *, minimum_outcome_delay_hours: int = 1
) -> tuple[list[TaskCandidate], FlywheelAudit]:
    """Filter and aggregate synthetic failures without retaining raw session identifiers.

    This function discovers task *candidates*. It does not train on production sessions, generate
    executable environments, or move candidates into a fixed evaluation set. A human must still
    reconstruct a licensed, deterministic task and run the normal security/review process.
    """

    if minimum_outcome_delay_hours < 0:
        raise ValueError("minimum outcome delay must be nonnegative")
    counters: Counter[str] = Counter()
    accepted: dict[tuple[str, str], list[str]] = {}
    for signal in signals:
        if not signal.research_consent:
            counters["no_consent"] += 1
        elif signal.contains_sensitive_data:
            counters["sensitive"] += 1
        elif signal.outcome_delay_hours < minimum_outcome_delay_hours:
            counters["unresolved"] += 1
        elif signal.outcome_success:
            counters["success"] += 1
        elif signal.in_fixed_evaluation:
            counters["fixed_evaluation"] += 1
        else:
            accepted.setdefault((signal.task_family, signal.failure_code), []).append(
                signal.session_id
            )

    candidates = [
        TaskCandidate(
            task_family=task_family,
            failure_code=failure_code,
            observations=len(session_ids),
            source_ids_sha256=hashlib.sha256(
                "\n".join(sorted(session_ids)).encode("utf-8")
            ).hexdigest(),
        )
        for (task_family, failure_code), session_ids in sorted(accepted.items())
    ]
    return candidates, FlywheelAudit(
        input_signals=len(signals),
        accepted_failure_signals=sum(candidate.observations for candidate in candidates),
        rejected_no_consent=counters["no_consent"],
        rejected_sensitive=counters["sensitive"],
        rejected_unresolved=counters["unresolved"],
        rejected_success=counters["success"],
        rejected_fixed_evaluation=counters["fixed_evaluation"],
    )
