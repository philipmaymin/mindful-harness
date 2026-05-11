"""Drift detection.

Long-running state has time-based and event-based frame-checks. As the
firehose continues and the Mind's beliefs accumulate, the original
framings drift away from the current evidence. Drift detection fires
on a schedule and asks: "Is what we agreed an hour ago still operative?"

The detector here is the structural piece: it tracks ages, knows what
to flag, and can be queried by the Mind. The actual reframing (what to
do with a stale belief) is a separate step — usually a Hand spawned by
the Mind to investigate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mindful_harness.mind import Mind


@dataclass
class DriftReport:
    """A snapshot of which Mind-state items are stale.

    A stale belief is one whose framing has not been re-examined for
    longer than the threshold. Stale doesn't mean wrong; it means
    untouched. The Mind decides whether to act.
    """

    stale_beliefs: list[tuple[str, float]]  # (key, age in seconds)
    stale_knowledge: list[tuple[str, float]]
    stale_decisions: list[tuple[str, float]]
    aging_questions: list[tuple[str, float]]
    overall_oldest: float = 0.0


def detect_drift(
    mind: Mind,
    belief_threshold_seconds: float = 86400.0,  # 24 hours
    knowledge_threshold_seconds: float = 7 * 86400.0,  # 7 days
    decision_threshold_seconds: float = 3 * 86400.0,  # 3 days
    question_threshold_seconds: float = 14 * 86400.0,  # 14 days
) -> DriftReport:
    """Scan the Mind for state that has aged past its threshold.

    Beliefs drift fastest (they're working models); knowledge drifts
    slower (it has provenance); decisions in the middle. Open
    questions that linger past the threshold get flagged separately —
    a question the Mind has held for two weeks without progress is
    itself information.

    Note: Conditionals do not currently carry a `last_reviewed`
    timestamp. v0.0.x uses item position in the dict as a proxy for
    age — newer items appear later. A dedicated timestamp field will
    land when the Mind starts re-examining beliefs in place.
    """
    now = time.time()

    stale_beliefs: list[tuple[str, float]] = []
    for key, c in mind.beliefs.items():
        age = now - c.last_reviewed
        if age > belief_threshold_seconds:
            stale_beliefs.append((key, age))

    stale_knowledge: list[tuple[str, float]] = []
    for key, c in mind.knowledge.items():
        age = now - c.last_reviewed
        if age > knowledge_threshold_seconds:
            stale_knowledge.append((key, age))

    stale_decisions: list[tuple[str, float]] = []
    for d in mind.decisions:
        age = now - d.timestamp
        if age > decision_threshold_seconds:
            stale_decisions.append((str(d.chosen), age))

    aging_questions: list[tuple[str, float]] = []
    for q in mind.questions:
        if q.resolved:
            continue
        age = now - q.born_at
        if age > question_threshold_seconds:
            aging_questions.append((q.text, age))

    candidates = (
        [age for _, age in stale_beliefs]
        + [age for _, age in stale_knowledge]
        + [age for _, age in stale_decisions]
        + [age for _, age in aging_questions]
    )
    overall_oldest = max(candidates, default=0.0)

    return DriftReport(
        stale_beliefs=stale_beliefs,
        stale_knowledge=stale_knowledge,
        stale_decisions=stale_decisions,
        aging_questions=aging_questions,
        overall_oldest=overall_oldest,
    )
