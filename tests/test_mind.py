"""Tests for the Mind layer."""

import time

import pytest

from mindful_harness import (
    Conditional,
    Decision,
    Distinction,
    FirehoseItem,
    Mind,
)


class TestMindIngestion:
    def test_ingest_appends_to_log(self) -> None:
        m = Mind()
        m.ingest(FirehoseItem(source="email", content="hello"))
        assert len(m.ingestion_log) == 1
        assert m.ingestion_log[0].source == "email"

    def test_multiple_ingests_preserve_order(self) -> None:
        m = Mind()
        for i in range(5):
            m.ingest(FirehoseItem(source="rss", content=f"item-{i}"))
        contents = [item.content for item in m.ingestion_log]
        assert contents == [f"item-{i}" for i in range(5)]


class TestMindBeliefs:
    def test_believe_stores_conditional(self) -> None:
        m = Mind()
        c = Conditional(value="Q1 up 12%", confidence=0.7, alternatives=["flat", "down"])
        m.believe("revenue", c)
        assert "revenue" in m.beliefs
        assert m.beliefs["revenue"].value == "Q1 up 12%"

    def test_believe_overwrites(self) -> None:
        m = Mind()
        m.believe("x", Conditional(value="A", confidence=0.5, alternatives=["B", "C"]))
        m.believe("x", Conditional(value="D", confidence=0.6, alternatives=["E", "F"]))
        assert m.beliefs["x"].value == "D"


class TestMindQuestions:
    def test_ask_appends(self) -> None:
        m = Mind()
        q = m.ask("why did Tuesday cohort retain 3x better than Monday?")
        assert q in m.questions
        assert not q.resolved

    def test_ask_with_routing(self) -> None:
        m = Mind()
        q = m.ask("when does the migration finish?", routed_to="hands")
        assert q.routed_to == "hands"


class TestMindInterests:
    def test_notice_creates_interest(self) -> None:
        m = Mind()
        i = m.notice("support load dropped while NPS rose")
        assert i in m.interests
        assert i.intensity == pytest.approx(1.0)

    def test_repeat_notice_reinforces_existing(self) -> None:
        m = Mind()
        m.notice("anomaly X")
        before = len(m.interests)
        i = m.notice("anomaly X")
        assert len(m.interests) == before
        assert i.intensity == pytest.approx(1.0)

    def test_decay_lowers_intensity(self) -> None:
        m = Mind()
        i = m.notice("decaying")
        i.last_reinforced = time.time() - 86400.0  # one day ago, half-life
        i.decay(half_life_seconds=86400.0)
        assert 0.4 < i.intensity < 0.6


class TestMindHabituation:
    def test_habituation_alarm_at_three(self) -> None:
        m = Mind()
        sig = "send-weekly-report"
        assert not m.is_habituated(sig)
        m.record_action(sig)
        m.record_action(sig)
        assert not m.is_habituated(sig)
        m.record_action(sig)
        assert m.is_habituated(sig)

    def test_record_action_returns_count(self) -> None:
        m = Mind()
        assert m.record_action("a") == 1
        assert m.record_action("a") == 2
        assert m.record_action("b") == 1


class TestMindCertaintyAlarms:
    def test_no_alarms_below_threshold(self) -> None:
        m = Mind()
        m.believe("x", Conditional(value="A", confidence=0.5, alternatives=["B", "C"]))
        assert m.certainty_alarms() == []

    def test_alarm_above_threshold(self) -> None:
        m = Mind()
        m.believe(
            "x", Conditional(value="A", confidence=0.95, alternatives=["B", "C"])
        )
        alarms = m.certainty_alarms()
        assert len(alarms) == 1
        assert alarms[0][0] == "belief:x"

    def test_alarm_distinguishes_belief_and_knowledge(self) -> None:
        m = Mind()
        m.believe(
            "b", Conditional(value="X", confidence=0.95, alternatives=["Y", "Z"])
        )
        m.know("k", Conditional(value="X", confidence=0.95, alternatives=["Y", "Z"]))
        alarms = m.certainty_alarms()
        keys = {a[0] for a in alarms}
        assert keys == {"belief:b", "knowledge:k"}


class TestMindVitalSigns:
    def test_empty_mind_baseline(self) -> None:
        m = Mind()
        signs = m.vital_signs()
        assert signs["items_ingested"] == 0.0
        assert signs["alternative_cardinality"] == 0.0

    def test_alternative_cardinality_averages(self) -> None:
        m = Mind()
        m.believe(
            "a", Conditional(value="X", confidence=0.5, alternatives=["Y", "Z"])
        )  # 2 alternatives
        m.believe(
            "b",
            Conditional(value="X", confidence=0.5, alternatives=["Y", "Z", "W", "V"]),
        )  # 4 alternatives
        signs = m.vital_signs()
        assert signs["alternative_cardinality"] == pytest.approx(3.0)

    def test_open_questions_counted(self) -> None:
        m = Mind()
        m.ask("Q1")
        q = m.ask("Q2")
        q.resolved = True
        assert m.vital_signs()["open_questions"] == 1.0


class TestMindDecisionsAndDistinctions:
    def test_commit_appends_decision(self) -> None:
        m = Mind()
        d = Decision(
            chosen="ship-it",
            rejected=["delay", "cancel"],
            framing="rev-up cycle",
            reversion_triggers=["if churn > 5%"],
            confidence=0.7,
        )
        m.commit(d)
        assert d in m.decisions

    def test_record_distinction_appends(self) -> None:
        m = Mind()
        d = Distinction(item="A", compared_to="B", noticed="A is novel")
        m.record_distinction(d)
        assert d in m.distinction_log


class TestMindOpportunitiesAndIdeas:
    def test_see_opportunity(self) -> None:
        m = Mind()
        o = m.see_opportunity(
            text="vendor X exposed an API",
            enables="we can drop the workaround",
        )
        assert o in m.opportunities
        assert not o.acted_on

    def test_connect_creates_idea(self) -> None:
        m = Mind()
        i = m.connect(
            text="support drop / NPS rise looks like the chambermaid effect",
            connects=["customer-support", "chambermaid-study"],
        )
        assert i in m.ideas
        assert "chambermaid-study" in i.connects


class TestMindEndToEnd:
    """A full workflow exercising the seven epistemic categories.

    This is the spec made executable: a Mind that takes in a stream,
    believes things, knows things, notices anomalies, asks questions,
    sees opportunities, connects ideas, commits decisions, and reports
    vital signs and certainty alarms. No LLM yet — just the structure.
    """

    def test_founders_inbox_scenario(self) -> None:
        m = Mind()

        m.ingest(
            FirehoseItem(
                source="email",
                content="Vendor X announced an API for direct integration",
            )
        )
        m.ingest(
            FirehoseItem(source="dashboard", content="Q1 revenue +12% vs last year")
        )

        m.believe(
            "Q1 trend",
            Conditional(
                value="up 12%",
                confidence=0.7,
                alternatives=["flat with measurement noise", "lagged from Q4"],
                framing="under the assumption that the dashboard cache is fresh",
                reversion_trigger="if the data refresh failed silently",
            ),
        )
        m.know(
            "vendor-X-api-launched",
            Conditional(
                value=True,
                confidence=0.95,
                alternatives=[False, "rumor not confirmed"],
            ),
        )

        m.notice("Q1 revenue moved despite no marketing change")
        m.ask("what drove Q1 revenue?", routed_to="hands")
        m.see_opportunity(
            text="Vendor X API enables direct integration",
            enables="we can deprecate the workaround layer",
        )
        m.connect(
            text="revenue moved without marketing change resembles the chambermaid effect",
            connects=["revenue-trend", "chambermaid-study"],
        )
        m.commit(
            Decision(
                chosen="investigate Q1 driver before acting",
                rejected=["assume seasonal", "assume measurement noise"],
                framing="initial Q1 review",
                reversion_triggers=[
                    "if dashboard cache proves stale",
                    "if Q2 reverses the trend",
                ],
                confidence=0.6,
            )
        )

        signs = m.vital_signs()
        assert signs["items_ingested"] == 2.0
        assert signs["open_questions"] == 1.0
        assert signs["active_interests"] == 1.0
        assert signs["opportunities_pending"] == 1.0
        assert signs["decisions_held"] == 1.0
        assert signs["alternative_cardinality"] >= 2.0  # both have alternatives

        alarms = m.certainty_alarms()
        assert any("vendor-X-api-launched" in name for name, _ in alarms)
        assert not any("Q1 trend" in name for name, _ in alarms)  # 0.7 < 0.9

    def test_habituation_in_workflow(self) -> None:
        m = Mind()
        for _ in range(3):
            m.ingest(FirehoseItem(source="cron", content="weekly report sent"))
            m.record_action("send-weekly-report")
        assert m.is_habituated("send-weekly-report")
