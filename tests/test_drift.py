"""Tests for drift detection."""

import time

from mindful_harness import Decision, Mind
from mindful_harness.drift import detect_drift


class TestDriftReport:
    def test_empty_mind_has_no_drift(self) -> None:
        report = detect_drift(Mind())
        assert report.stale_decisions == []
        assert report.aging_questions == []
        assert report.overall_oldest == 0.0

    def test_recent_decision_not_flagged(self) -> None:
        m = Mind()
        m.commit(
            Decision(
                chosen="A",
                rejected=["B", "C"],
                framing="recent",
                reversion_triggers=["X"],
                confidence=0.6,
            )
        )
        report = detect_drift(m)
        assert report.stale_decisions == []

    def test_old_decision_flagged(self) -> None:
        m = Mind()
        d = Decision(
            chosen="legacy-choice",
            rejected=["B", "C"],
            framing="months ago",
            reversion_triggers=["X"],
            confidence=0.6,
        )
        d.timestamp = time.time() - 10 * 86400.0  # 10 days ago
        m.commit(d)
        report = detect_drift(m, decision_threshold_seconds=3 * 86400.0)
        assert len(report.stale_decisions) == 1
        assert report.stale_decisions[0][0] == "legacy-choice"

    def test_threshold_configurable(self) -> None:
        m = Mind()
        d = Decision(
            chosen="A",
            rejected=["B", "C"],
            framing="f",
            reversion_triggers=["X"],
            confidence=0.5,
        )
        d.timestamp = time.time() - 60.0  # 1 minute old
        m.commit(d)
        report_lenient = detect_drift(m, decision_threshold_seconds=300.0)
        assert report_lenient.stale_decisions == []
        report_strict = detect_drift(m, decision_threshold_seconds=30.0)
        assert len(report_strict.stale_decisions) == 1

    def test_aging_question_flagged(self) -> None:
        m = Mind()
        q = m.ask("why does the Tuesday cohort retain better?")
        q.born_at = time.time() - 20 * 86400.0
        report = detect_drift(m, question_threshold_seconds=14 * 86400.0)
        assert len(report.aging_questions) == 1

    def test_resolved_question_not_flagged(self) -> None:
        m = Mind()
        q = m.ask("Q?")
        q.born_at = time.time() - 30 * 86400.0
        q.resolved = True
        report = detect_drift(m, question_threshold_seconds=14 * 86400.0)
        assert report.aging_questions == []

    def test_overall_oldest_reported(self) -> None:
        m = Mind()
        old_decision = Decision(
            chosen="A",
            rejected=["B", "C"],
            framing="f",
            reversion_triggers=["X"],
            confidence=0.5,
        )
        old_decision.timestamp = time.time() - 100 * 86400.0
        m.commit(old_decision)
        q = m.ask("aging?")
        q.born_at = time.time() - 50 * 86400.0
        report = detect_drift(m)
        # overall_oldest should be at least 50 days
        assert report.overall_oldest > 50 * 86400.0
