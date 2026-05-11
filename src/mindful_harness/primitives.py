"""Structural primitives of the Mindful Harness.

Each primitive is grounded in a specific Langer technique and enforced
at the type level rather than through prompting. The first primitive
shipped here is `Conditional[T]`, which encodes "could be" rather than
"is" and structurally requires three or more framings.

See SPEC.md for the full primitive list and the Langer sources.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


CERTAINTY_THRESHOLD: float = 0.9
"""Confidence at or above this level fires the certainty alarm.

Langer (Sept 18 transcript): 'When you hear yourself or feel yourself
knowing something with certainty, that's the signal of mindlessness.'
"""


@dataclass
class Conditional(Generic[T]):
    """A value held conditionally rather than absolutely.

    Encodes Langer's 'could be' versus 'is' intervention. Carries the
    value, its confidence, explicit alternatives, the framing under
    which it holds, the condition that would trigger revision, and the
    timestamp at which the value was last reviewed (used by drift
    detection to flag aging claims).

    Constructing one with fewer than three total framings is a type
    error, not a runtime warning, because mindlessness creeps in
    silently when single-framed values propagate as facts.
    """

    value: T
    confidence: float
    alternatives: list[T] = field(default_factory=list)
    framing: str = ""
    reversion_trigger: str | None = None
    last_reviewed: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        total_framings = 1 + len(self.alternatives)
        if total_framings < 3:
            raise ValueError(
                "Conditional requires at least three framings "
                f"(value + alternatives), got {total_framings}. "
                "Three framings invite four, five, six; two collapses to a binary."
            )

    def __str__(self) -> str:
        return f"could be {self.value!r} (confidence {self.confidence:.2f})"

    def touch(self) -> None:
        """Mark this Conditional as freshly reviewed (resets the drift clock)."""
        self.last_reviewed = time.time()


def is_certain(c: Conditional[T]) -> bool:
    """Return True when a Conditional has crossed the certainty threshold.

    A True return is a signal to interrogate, not to trust. The harness
    treats certainty as an alarm rather than as a feature.
    """
    return c.confidence >= CERTAINTY_THRESHOLD


@dataclass
class Decision(Generic[T]):
    """A commitment held loosely.

    Encodes the decisions-held-loosely primitive: every commitment carries
    the conditions under which it would be revisited. Without explicit
    reversion triggers, decisions ossify into invisible commitments and
    the agent stops noticing evidence that should reverse them.
    Constructing a Decision without triggers, or with fewer than two
    rejected alternatives, is a type error.
    """

    chosen: T
    rejected: list[T]
    framing: str
    reversion_triggers: list[str]
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        if not self.reversion_triggers:
            raise ValueError(
                "Decision requires at least one reversion trigger. "
                "Decisions without triggers ossify silently."
            )
        if len(self.rejected) < 2:
            raise ValueError(
                "Decision requires at least two rejected alternatives "
                f"(three options total), got {len(self.rejected)}. "
                "Three options ensures the choice was a choice, not a default."
            )

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def __str__(self) -> str:
        return (
            f"chose {self.chosen!r} over {self.rejected!r} "
            f"under framing {self.framing!r}, "
            f"revisits if: {'; '.join(self.reversion_triggers)}"
        )


@dataclass
class Distinction:
    """An explicit distinction noted between two items.

    Active distinction-making is the operational antidote to passive
    pattern-matching. The Mind notices what is different from prior
    items in the same category, what is novel, what might force a new
    category. A Distinction records the comparison and the difference
    that was noticed, with an empty difference being a type error
    rather than a default — passive attention is not mindfulness.
    """

    item: Any
    compared_to: Any
    noticed: str
    forces_new_category: bool = False

    def __post_init__(self) -> None:
        if not self.noticed.strip():
            raise ValueError(
                "Distinction requires an explicit noticed-difference. "
                "Empty distinctions are passive attention, not mindfulness."
            )

    def __str__(self) -> str:
        marker = " [forces new category]" if self.forces_new_category else ""
        return f"noticed: {self.noticed}{marker}"
