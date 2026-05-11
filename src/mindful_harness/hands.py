"""The Hands: spawnable agents for episodic action.

When the Mind notices something action-worthy (high-stakes opportunity,
user request, certainty alarm needing investigation), it spawns a Hand.
Each Hand inherits relevant Mind state at spawn time, executes its task
using the Langer primitives at the per-action level, and returns a
HandResult that the Mind can integrate.

The structure here (Hand, Advocate, FrameworkQuestioner) is scaffolded
so the orchestration shape is testable. Execution methods default to
manual-input pathways so the harness can be exercised without an LLM;
LLM-driven execution lands as a separate adapter.
"""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mindful_harness.mind import Mind
from mindful_harness.primitives import Decision


@dataclass
class HandResult:
    """The output of a Hand's work.

    Always carries multiple framings (per the chambermaid effect from
    Mindful Body p.42): the receiver picks which framing they will act
    under, knowing it is a choice rather than a default. A result with
    fewer than three framings is a type error.

    `would_revise_if` carries the LLM-produced reversion triggers (the
    "decisions held loosely" primitive at the per-action level); these
    flow into `Decision.reversion_triggers` when the result is
    committed to the Mind.
    """

    chosen: Any
    rejected_alternatives: list[Any]
    framings: dict[str, str]
    framework: str
    confidence: float
    process_trail: list[str] = field(default_factory=list)
    advocate_critique: str | None = None
    questioner_challenges: list[str] = field(default_factory=list)
    would_revise_if: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        if len(self.framings) < 3:
            raise ValueError(
                "HandResult requires at least three framings "
                f"(per chambermaid effect), got {len(self.framings)}. "
                "The receiver should pick which framing to act under."
            )
        if not self.framework.strip():
            raise ValueError(
                "HandResult requires explicit framework. "
                "Invisible frameworks are how the expert trap installs itself."
            )

    def to_decision(self, reversion_triggers: list[str] | None = None) -> Decision:
        """Convert to a Decision for committing to the Mind.

        Defaults reversion_triggers to whatever the Hand itself surfaced
        in `would_revise_if`, so a Decision committed from a HandResult
        is held loosely by default.
        """
        triggers = list(reversion_triggers) if reversion_triggers else list(self.would_revise_if)
        if not triggers:
            raise ValueError(
                "Decision requires at least one reversion trigger. "
                "Pass reversion_triggers explicitly or have the Hand "
                "produce would_revise_if."
            )
        return Decision(
            chosen=self.chosen,
            rejected=list(self.rejected_alternatives),
            framing=self.framework,
            reversion_triggers=triggers,
            confidence=self.confidence,
        )


@dataclass
class Hand:
    """A single agent spawned for episodic action.

    The Hand is stateless across spawns (it gets fresh Mind state each
    time) but stateful within a spawn (its process trail accumulates).
    A Hand executes a task, takes a critique from a mindful advocate,
    takes challenges from a framework questioner, and only then
    returns a HandResult.
    """

    id: str
    task: str
    mind_snapshot: dict[str, Any]
    framework: str
    spawned_at: float = field(default_factory=time.time)
    process_trail: list[str] = field(default_factory=list)

    def log(self, note: str) -> None:
        """Append a step to the process trail."""
        self.process_trail.append(f"[{time.time() - self.spawned_at:.2f}s] {note}")

    def execute_manual(
        self,
        chosen: Any,
        rejected_alternatives: list[Any],
        framings: dict[str, str],
        confidence: float,
        would_revise_if: list[str] | None = None,
    ) -> HandResult:
        """Produce a HandResult from explicit caller input.

        This is the LLM-free path: a human or a deterministic procedure
        produces what the LLM would produce. Useful for testing the
        orchestration shape and for high-stakes decisions where the
        human wants to drive directly.

        Pass `would_revise_if` so the resulting HandResult can be
        converted to a Decision via `to_decision()` without supplying
        reversion triggers separately.
        """
        self.log(f"manual execution: chose {chosen!r}")
        return HandResult(
            chosen=chosen,
            rejected_alternatives=rejected_alternatives,
            framings=framings,
            framework=self.framework,
            confidence=confidence,
            process_trail=list(self.process_trail),
            would_revise_if=list(would_revise_if) if would_revise_if else [],
        )


@dataclass
class Advocate:
    """Disagrees mindfully with the proposed action.

    Separately instantiated from the worker Hand. Its job is to
    surface the strongest version of the disagreement: with multiple
    framings, conditional language, openness to being wrong itself,
    and concrete reasoning. The output is a string critique with at
    least one specific concern.
    """

    proposal: HandResult

    def critique_manual(self, critique: str) -> str:
        """LLM-free path: caller provides the critique directly."""
        if not critique.strip():
            raise ValueError(
                "Advocate critique cannot be empty. "
                "An empty disagreement is performative debate."
            )
        return critique


@dataclass
class FrameworkQuestioner:
    """Questions the framework itself, not the action.

    Distinct from the advocate: the advocate challenges what to do;
    the questioner challenges the lens through which the decision is
    being framed. Output is a list of at least one challenge to the
    operating framework.
    """

    framework: str

    def question_manual(self, challenges: list[str]) -> list[str]:
        """LLM-free path: caller provides framework challenges directly."""
        if not challenges:
            raise ValueError(
                "FrameworkQuestioner requires at least one challenge. "
                "Frameworks unchallenged remain invisible."
            )
        for c in challenges:
            if not c.strip():
                raise ValueError("Empty challenge: framework questioning is vacuous.")
        return challenges


def spawn_hand(
    mind: Mind,
    task: str,
    framework: str,
    snapshot_keys: list[str] | None = None,
) -> Hand:
    """Spawn a Hand with relevant Mind state deep-copied at spawn time.

    The snapshot is a deep copy: the Hand operates on what the Mind
    believed when the work began, and subsequent Mind mutations cannot
    reach into the running Hand's view. If new evidence arrives during
    execution, the Mind will note it and may issue a new Hand, but the
    running one isn't disrupted mid-flight.
    """
    snapshot: dict[str, Any] = {
        "spawned_at": time.time(),
        "beliefs": copy.deepcopy(mind.beliefs),
        "knowledge": copy.deepcopy(mind.knowledge),
        "open_questions": [q.text for q in mind.questions if not q.resolved],
        "interests": [i.text for i in mind.interests],
    }
    if snapshot_keys is not None:
        snapshot = {k: snapshot[k] for k in snapshot_keys if k in snapshot}

    return Hand(
        id=str(uuid.uuid4())[:8],
        task=task,
        mind_snapshot=snapshot,
        framework=framework,
    )


def run_with_advocate_and_questioner(
    hand: Hand,
    execute: Callable[[Hand], HandResult],
    advocate_critique: Callable[[HandResult], str],
    framework_challenges: Callable[[str], list[str]],
) -> HandResult:
    """Run a Hand with mandatory advocate critique and framework questioning.

    This is the structural commitment from the spec: debate is not
    optional, framework questioning is not optional. The function
    composes a Hand's execution with both, returning a HandResult
    whose advocate_critique and questioner_challenges are populated.
    """
    result = execute(hand)
    hand.log("execution complete; consulting mindful advocate")
    critique = advocate_critique(result)
    hand.log("advocate critique received; consulting framework questioner")
    challenges = framework_challenges(hand.framework)
    hand.log("framework questioning complete")

    return HandResult(
        chosen=result.chosen,
        rejected_alternatives=result.rejected_alternatives,
        framings=result.framings,
        framework=result.framework,
        confidence=result.confidence,
        process_trail=list(hand.process_trail),
        advocate_critique=critique,
        questioner_challenges=challenges,
        would_revise_if=list(result.would_revise_if),
    )
