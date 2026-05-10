"""Tests for the structural primitives."""

import time

import pytest

from mindful_harness.primitives import (
    CERTAINTY_THRESHOLD,
    Conditional,
    Decision,
    Distinction,
    KindfulnessVector,
    is_certain,
)


class TestConditional:
    def test_three_framings_is_minimum(self) -> None:
        with pytest.raises(ValueError, match="three framings"):
            Conditional(value="X", confidence=0.5, alternatives=[])

    def test_two_framings_rejected(self) -> None:
        with pytest.raises(ValueError, match="three framings"):
            Conditional(value="X", confidence=0.5, alternatives=["Y"])

    def test_three_framings_accepted(self) -> None:
        c = Conditional(value="X", confidence=0.5, alternatives=["Y", "Z"])
        assert c.value == "X"
        assert len(c.alternatives) == 2

    def test_more_than_three_framings_accepted(self) -> None:
        c = Conditional(value="X", confidence=0.5, alternatives=["Y", "Z", "W", "V"])
        assert len(c.alternatives) == 4

    def test_confidence_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Conditional(value="X", confidence=1.5, alternatives=["Y", "Z"])
        with pytest.raises(ValueError, match="confidence"):
            Conditional(value="X", confidence=-0.1, alternatives=["Y", "Z"])

    def test_confidence_endpoints_accepted(self) -> None:
        Conditional(value="X", confidence=0.0, alternatives=["Y", "Z"])
        Conditional(value="X", confidence=1.0, alternatives=["Y", "Z"])

    def test_str_uses_could_be(self) -> None:
        c = Conditional(value="alpha", confidence=0.7, alternatives=["beta", "gamma"])
        text = str(c)
        assert "could be" in text
        assert "alpha" in text
        assert "is " not in text

    def test_reversion_trigger_optional(self) -> None:
        c = Conditional(value="X", confidence=0.5, alternatives=["Y", "Z"])
        assert c.reversion_trigger is None
        c2 = Conditional(
            value="X",
            confidence=0.5,
            alternatives=["Y", "Z"],
            reversion_trigger="if metric drops below 0.3",
        )
        assert c2.reversion_trigger == "if metric drops below 0.3"

    def test_framing_optional(self) -> None:
        c = Conditional(value="X", confidence=0.5, alternatives=["Y", "Z"])
        assert c.framing == ""
        c2 = Conditional(
            value="X",
            confidence=0.5,
            alternatives=["Y", "Z"],
            framing="under the assumption that the data is recent",
        )
        assert "recent" in c2.framing


class TestCertaintyAlarm:
    def test_high_confidence_triggers_alarm(self) -> None:
        c = Conditional(value="X", confidence=0.95, alternatives=["Y", "Z"])
        assert is_certain(c)

    def test_low_confidence_no_alarm(self) -> None:
        c = Conditional(value="X", confidence=0.4, alternatives=["Y", "Z"])
        assert not is_certain(c)

    def test_threshold_boundary(self) -> None:
        at_threshold = Conditional(
            value="X", confidence=CERTAINTY_THRESHOLD, alternatives=["Y", "Z"]
        )
        assert is_certain(at_threshold)
        below_threshold = Conditional(
            value="X", confidence=CERTAINTY_THRESHOLD - 0.001, alternatives=["Y", "Z"]
        )
        assert not is_certain(below_threshold)


class TestDecision:
    def test_requires_reversion_trigger(self) -> None:
        with pytest.raises(ValueError, match="reversion trigger"):
            Decision(
                chosen="A",
                rejected=["B", "C"],
                framing="under framing X",
                reversion_triggers=[],
                confidence=0.7,
            )

    def test_requires_two_rejected_alternatives(self) -> None:
        with pytest.raises(ValueError, match="rejected alternatives"):
            Decision(
                chosen="A",
                rejected=[],
                framing="X",
                reversion_triggers=["if Y happens"],
                confidence=0.7,
            )
        with pytest.raises(ValueError, match="rejected alternatives"):
            Decision(
                chosen="A",
                rejected=["B"],
                framing="X",
                reversion_triggers=["if Y happens"],
                confidence=0.7,
            )

    def test_three_options_accepted(self) -> None:
        d = Decision(
            chosen="A",
            rejected=["B", "C"],
            framing="under framing X",
            reversion_triggers=["if metric drops below 0.5"],
            confidence=0.7,
        )
        assert d.chosen == "A"
        assert len(d.rejected) == 2

    def test_confidence_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Decision(
                chosen="A",
                rejected=["B", "C"],
                framing="X",
                reversion_triggers=["Y"],
                confidence=2.0,
            )

    def test_age_increases(self) -> None:
        d = Decision(
            chosen="A",
            rejected=["B", "C"],
            framing="X",
            reversion_triggers=["Y"],
            confidence=0.5,
        )
        time.sleep(0.01)
        assert d.age_seconds() > 0

    def test_str_describes_decision(self) -> None:
        d = Decision(
            chosen="ship-it",
            rejected=["delay", "cancel"],
            framing="rev-up cycle",
            reversion_triggers=["if churn > 5%"],
            confidence=0.7,
        )
        text = str(d)
        assert "ship-it" in text
        assert "delay" in text
        assert "rev-up" in text
        assert "churn" in text


class TestKindfulnessVector:
    def test_requires_explicit_counterpart(self) -> None:
        with pytest.raises(ValueError, match="counterpart"):
            KindfulnessVector(toward="", disposition="generous")
        with pytest.raises(ValueError, match="counterpart"):
            KindfulnessVector(toward="   ", disposition="generous")

    def test_requires_explicit_disposition(self) -> None:
        with pytest.raises(ValueError, match="disposition"):
            KindfulnessVector(toward="the customer", disposition="")

    def test_well_formed_vector(self) -> None:
        v = KindfulnessVector(
            toward="the support team",
            disposition="curious and supportive",
            preserve_agency=True,
        )
        assert v.preserve_agency
        text = str(v)
        assert "support team" in text
        assert "curious" in text
        assert "preserving agency" in text

    def test_default_preserves_agency(self) -> None:
        v = KindfulnessVector(toward="X", disposition="Y")
        assert v.preserve_agency is True

    def test_explicit_agency_constraint_visible(self) -> None:
        v = KindfulnessVector(
            toward="adversary in negotiation",
            disposition="firm",
            preserve_agency=False,
        )
        assert "may constrain agency" in str(v)


class TestDistinction:
    def test_requires_noticed_difference(self) -> None:
        with pytest.raises(ValueError, match="noticed-difference"):
            Distinction(item="A", compared_to="B", noticed="")
        with pytest.raises(ValueError, match="noticed-difference"):
            Distinction(item="A", compared_to="B", noticed="   ")

    def test_well_formed_distinction(self) -> None:
        d = Distinction(
            item={"new": True},
            compared_to={"new": False},
            noticed="the 'new' flag flipped",
        )
        assert "new" in d.noticed
        assert "flipped" in str(d)

    def test_forces_new_category_marker(self) -> None:
        d = Distinction(
            item={"shape": "torus"},
            compared_to={"shape": "sphere"},
            noticed="topological genus differs",
            forces_new_category=True,
        )
        assert "forces new category" in str(d)

    def test_default_does_not_force_new_category(self) -> None:
        d = Distinction(
            item={"shape": "circle"},
            compared_to={"shape": "ellipse"},
            noticed="aspect ratio differs",
        )
        assert "forces new category" not in str(d)
