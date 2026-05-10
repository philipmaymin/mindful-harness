"""Tests for counter-adage surfacing."""

import pytest

from mindful_harness.counter_adages import CANONICAL_ADAGE_PAIRS, CounterAdageStore


class TestCounterAdageStore:
    def test_canonical_set_loaded(self) -> None:
        s = CounterAdageStore()
        assert len(s) == len(CANONICAL_ADAGE_PAIRS)

    def test_opposite_lookup_exact(self) -> None:
        s = CounterAdageStore()
        assert s.opposite_of("look before you leap") == "he who hesitates is lost"

    def test_opposite_lookup_symmetric(self) -> None:
        s = CounterAdageStore()
        assert s.opposite_of("he who hesitates is lost") == "look before you leap"

    def test_opposite_lookup_normalizes(self) -> None:
        s = CounterAdageStore()
        assert s.opposite_of("Look Before You Leap.") == "he who hesitates is lost"
        assert s.opposite_of("  look before you leap  ") == "he who hesitates is lost"

    def test_opposite_returns_none_when_unknown(self) -> None:
        s = CounterAdageStore()
        assert s.opposite_of("a stitch in time saves nine") is None

    def test_find_partial_in_text(self) -> None:
        s = CounterAdageStore()
        text = (
            "Given the risk profile, I'd say look before you leap on this one. "
            "We need more data."
        )
        hits = s.find_partial(text)
        adages = {a for a, _ in hits}
        assert "look before you leap" in adages

    def test_find_partial_returns_opposite(self) -> None:
        s = CounterAdageStore()
        text = "Many hands make light work, so let's parallelize."
        hits = s.find_partial(text)
        assert ("many hands make light work", "too many cooks spoil the broth") in hits

    def test_find_partial_empty_when_no_match(self) -> None:
        s = CounterAdageStore()
        assert s.find_partial("This text contains no folk wisdom whatsoever.") == []

    def test_add_extends_store(self) -> None:
        s = CounterAdageStore()
        before = len(s)
        s.add("ship fast and break things", "measure twice cut once")
        assert len(s) == before + 1
        assert s.opposite_of("ship fast and break things") == "measure twice cut once"

    def test_add_rejects_empty(self) -> None:
        s = CounterAdageStore()
        with pytest.raises(ValueError):
            s.add("", "non-empty")
        with pytest.raises(ValueError):
            s.add("non-empty", "")
