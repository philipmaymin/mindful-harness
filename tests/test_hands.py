"""Tests for the Hands layer."""

import pytest

from mindful_harness import (
    Advocate,
    Conditional,
    FrameworkQuestioner,
    Hand,
    HandResult,
    Mind,
    run_with_advocate_and_questioner,
    spawn_hand,
)


def _three_framings() -> dict[str, str]:
    return {
        "build-it": "treat the proposal as worth building this quarter",
        "park-it": "treat the proposal as worth tracking but not yet acting on",
        "kill-it": "treat the proposal as worth ending the conversation",
    }


class TestHandResult:
    def test_requires_three_framings(self) -> None:
        with pytest.raises(ValueError, match="three framings"):
            HandResult(
                chosen="A",
                rejected_alternatives=["B"],
                framings={"only-one": "x"},
                framework="test framework",
                confidence=0.5,
            )

    def test_requires_explicit_framework(self) -> None:
        with pytest.raises(ValueError, match="framework"):
            HandResult(
                chosen="A",
                rejected_alternatives=["B"],
                framings=_three_framings(),
                framework="",
                confidence=0.5,
            )

    def test_well_formed_result(self) -> None:
        r = HandResult(
            chosen="build-it",
            rejected_alternatives=["park-it", "kill-it"],
            framings=_three_framings(),
            framework="quarter-planning",
            confidence=0.6,
                    )
        assert r.chosen == "build-it"
        assert len(r.framings) == 3

    def test_to_decision(self) -> None:
        r = HandResult(
            chosen="ship-it",
            rejected_alternatives=["delay", "cancel"],
            framings=_three_framings(),
            framework="rev-up cycle",
            confidence=0.7,
        )
        d = r.to_decision(reversion_triggers=["if churn > 5%"])
        assert d.chosen == "ship-it"
        assert "churn" in d.reversion_triggers[0]


class TestSpawnHand:
    def test_snapshot_contains_beliefs(self) -> None:
        m = Mind()
        m.believe(
            "x", Conditional(value="A", confidence=0.5, alternatives=["B", "C"])
        )
        h = spawn_hand(
            m, task="investigate x", framework="initial"        )
        assert "beliefs" in h.mind_snapshot
        assert "x" in h.mind_snapshot["beliefs"]

    def test_snapshot_contains_open_questions(self) -> None:
        m = Mind()
        m.ask("why X?")
        h = spawn_hand(
            m, task="investigate x", framework="initial"        )
        assert "why X?" in h.mind_snapshot["open_questions"]

    def test_snapshot_keys_filter(self) -> None:
        m = Mind()
        m.believe(
            "x", Conditional(value="A", confidence=0.5, alternatives=["B", "C"])
        )
        m.ask("why?")
        h = spawn_hand(
            m,
            task="...",
            framework="...",
                        snapshot_keys=["beliefs"],
        )
        assert "beliefs" in h.mind_snapshot
        assert "open_questions" not in h.mind_snapshot

    def test_hand_has_unique_id(self) -> None:
        m = Mind()
        h1 = spawn_hand(m, task="t", framework="f")
        h2 = spawn_hand(m, task="t", framework="f")
        assert h1.id != h2.id

    def test_snapshot_is_deep_copy_not_shallow(self) -> None:
        """Mutating the Mind after spawn must not change the Hand's snapshot."""
        m = Mind()
        m.believe(
            "drift-target",
            Conditional(value="original", confidence=0.5, alternatives=["b", "c"]),
        )
        h = spawn_hand(m, task="t", framework="f")
        # Replace the belief in the live Mind with a fresh Conditional.
        m.believe(
            "drift-target",
            Conditional(value="mutated", confidence=0.9, alternatives=["x", "y"]),
        )
        snapshot_belief = h.mind_snapshot["beliefs"]["drift-target"]
        assert snapshot_belief.value == "original"
        assert snapshot_belief.confidence == 0.5


class TestHandManualExecute:
    def test_returns_hand_result(self) -> None:
        m = Mind()
        h = spawn_hand(m, task="t", framework="initial")
        r = h.execute_manual(
            chosen="A",
            rejected_alternatives=["B", "C"],
            framings=_three_framings(),
            confidence=0.6,
        )
        assert r.chosen == "A"
        assert r.framework == "initial"
        assert r.process_trail  # log appended

    def test_log_records_steps(self) -> None:
        m = Mind()
        h = spawn_hand(m, task="t", framework="f")
        h.log("starting work")
        h.log("noticed something")
        assert len(h.process_trail) == 2
        assert "starting work" in h.process_trail[0]

    def test_execute_manual_accepts_would_revise_if(self) -> None:
        """A manually-executed Hand should be convertible to a Decision."""
        m = Mind()
        h = spawn_hand(m, task="t", framework="initial")
        r = h.execute_manual(
            chosen="A",
            rejected_alternatives=["B", "C"],
            framings={"x": "...", "y": "...", "z": "..."},
            confidence=0.6,
            would_revise_if=["if metric drops below 0.3"],
        )
        d = r.to_decision()  # no explicit triggers needed; uses Hand's
        assert "metric drops" in d.reversion_triggers[0]


class TestAdvocate:
    def test_rejects_empty_critique(self) -> None:
        result = HandResult(
            chosen="A",
            rejected_alternatives=["B", "C"],
            framings=_three_framings(),
            framework="f",
            confidence=0.5,
        )
        adv = Advocate(proposal=result)
        with pytest.raises(ValueError, match="empty"):
            adv.critique_manual("")
        with pytest.raises(ValueError, match="empty"):
            adv.critique_manual("   ")

    def test_accepts_substantive_critique(self) -> None:
        result = HandResult(
            chosen="A",
            rejected_alternatives=["B", "C"],
            framings=_three_framings(),
            framework="f",
            confidence=0.5,
        )
        adv = Advocate(proposal=result)
        c = adv.critique_manual(
            "The chosen path assumes Q1 trend continues. If Q2 reverses, this locks us into the wrong path."
        )
        assert "Q2" in c


class TestFrameworkQuestioner:
    def test_rejects_empty_challenge_list(self) -> None:
        q = FrameworkQuestioner(framework="quarter-planning")
        with pytest.raises(ValueError, match="at least one"):
            q.question_manual([])

    def test_rejects_empty_individual_challenge(self) -> None:
        q = FrameworkQuestioner(framework="f")
        with pytest.raises(ValueError, match="vacuous"):
            q.question_manual(["valid one", ""])

    def test_accepts_substantive_challenges(self) -> None:
        q = FrameworkQuestioner(framework="quarter-planning")
        challenges = q.question_manual(
            [
                "Why are we framing this as a quarter, when the technical scope is 3 weeks?",
                "Quarter-planning assumes a quarter exists; a continuous-deployment culture has no quarters.",
            ]
        )
        assert len(challenges) == 2


class TestRunWithAdvocateAndQuestioner:
    def test_composes_full_pipeline(self) -> None:
        m = Mind()
        h = spawn_hand(
            m,
            task="decide on Q1 investigation",
            framework="initial Q1 review",
                    )

        def execute(hand: Hand) -> HandResult:
            return hand.execute_manual(
                chosen="investigate",
                rejected_alternatives=["assume seasonal", "assume noise"],
                framings=_three_framings(),
                confidence=0.6,
            )

        def advocate(result: HandResult) -> str:
            return "What if the Q1 lift is real and we're overthinking it?"

        def challenge(framework: str) -> list[str]:
            return [
                f"What if {framework!r} is the wrong frame, and this is a Q4 lag effect?"
            ]

        final = run_with_advocate_and_questioner(h, execute, advocate, challenge)
        assert final.advocate_critique
        assert final.questioner_challenges
        assert len(final.process_trail) >= 3  # log + advocate + questioner steps
