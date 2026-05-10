"""Tests for the Mind visualization."""

from __future__ import annotations

import json

import pytest

from mindful_harness import Conditional, Decision, Mind
from mindful_harness.viz import render_mind_html, render_mind_json


@pytest.fixture
def populated_mind() -> Mind:
    m = Mind()
    m.believe(
        "revenue trend",
        Conditional(
            value="Q1 up 12%",
            confidence=0.7,
            alternatives=["flat with noise", "Q4 carryover"],
            framing="under the assumption the dashboard cache is fresh",
        ),
    )
    m.know(
        "vendor announced API",
        Conditional(
            value=True,
            confidence=0.95,
            alternatives=[False, "rumor"],
        ),
    )
    m.ask("what drove Q1 revenue?")
    m.notice("support load dropped while NPS rose")
    m.wonder("how does Tuesday cohort differ from Monday?")
    m.see_opportunity(
        text="API enables direct integration",
        enables="we can deprecate the workaround",
    )
    m.connect(
        text="this resembles the chambermaid effect",
        connects=["revenue", "chambermaid-study"],
    )
    m.commit(
        Decision(
            chosen="investigate before acting",
            rejected=["assume seasonal", "assume noise"],
            framing="Q1 review",
            reversion_triggers=["if Q2 reverses"],
            confidence=0.6,
        )
    )
    return m


class TestRenderHTML:
    def test_renders_for_empty_mind(self) -> None:
        html = render_mind_html(Mind())
        assert "<!doctype html>" in html
        assert "no beliefs yet" in html
        assert "no decisions committed" in html

    def test_includes_belief_values(self, populated_mind: Mind) -> None:
        html = render_mind_html(populated_mind)
        assert "Q1 up 12%" in html
        assert "could be" in html

    def test_includes_alternatives(self, populated_mind: Mind) -> None:
        html = render_mind_html(populated_mind)
        assert "flat with noise" in html
        assert "Q4 carryover" in html

    def test_includes_certainty_alarm(self, populated_mind: Mind) -> None:
        html = render_mind_html(populated_mind)
        assert "Certainty alarms" in html
        assert "vendor announced API" in html

    def test_includes_opportunity_enables(self, populated_mind: Mind) -> None:
        html = render_mind_html(populated_mind)
        assert "enables:" in html
        assert "deprecate the workaround" in html

    def test_includes_idea_connects(self, populated_mind: Mind) -> None:
        html = render_mind_html(populated_mind)
        assert "chambermaid-study" in html
        assert "connects:" in html

    def test_includes_decision_with_reversion(self, populated_mind: Mind) -> None:
        html = render_mind_html(populated_mind)
        assert "investigate before acting" in html
        assert "if Q2 reverses" in html

    def test_includes_vital_signs(self, populated_mind: Mind) -> None:
        html = render_mind_html(populated_mind)
        assert "Vital signs" in html
        assert "items ingested" in html or "alternative cardinality" in html

    def test_escapes_html(self) -> None:
        m = Mind()
        m.believe(
            "danger",
            Conditional(
                value="<script>alert(1)</script>",
                confidence=0.5,
                alternatives=["safe", "neutral"],
            ),
        )
        html = render_mind_html(m)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_no_em_dashes_in_user_facing_copy(self) -> None:
        # Per project writing-style rules; check the whole document body and head.
        html = render_mind_html(Mind())
        assert "—" not in html


class TestRenderJSON:
    def test_empty_mind_returns_valid_json(self) -> None:
        text = render_mind_json(Mind())
        data = json.loads(text)
        assert data["beliefs"] == {}
        assert data["questions"] == []

    def test_populated_mind_json_structure(self, populated_mind: Mind) -> None:
        text = render_mind_json(populated_mind)
        data = json.loads(text)
        assert "revenue trend" in data["beliefs"]
        assert data["beliefs"]["revenue trend"]["value"] == "Q1 up 12%"
        assert data["beliefs"]["revenue trend"]["confidence"] == 0.7
        assert len(data["beliefs"]["revenue trend"]["alternatives"]) == 2

    def test_decisions_serialized(self, populated_mind: Mind) -> None:
        text = render_mind_json(populated_mind)
        data = json.loads(text)
        assert len(data["decisions"]) == 1
        d = data["decisions"][0]
        assert d["chosen"] == "investigate before acting"
        assert "Q2 reverses" in d["reversion_triggers"][0]

    def test_certainty_alarms_serialized(self, populated_mind: Mind) -> None:
        text = render_mind_json(populated_mind)
        data = json.loads(text)
        assert len(data["certainty_alarms"]) == 1
        assert "vendor announced API" in data["certainty_alarms"][0]["name"]
