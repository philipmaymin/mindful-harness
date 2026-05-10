"""Counter-adage surfacing.

For every heuristic the agent invokes ("look before you leap"), the
harness surfaces its opposite ("he who hesitates is lost"). Folk
wisdom that goes unchallenged is the most common mindlessness vector
in LLM training data.

The store ships with a small canonical set drawn from the book and
common English counter-adage pairs. Callers can extend it; an LLM
adapter for novel adages lands separately.

Source: book.md line 1162 (Mindful Learning chapter): "Has anyone
pointed out that 'look before you leap' and 'he who hesitates is
lost' can't both be laws of the universe?"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CANONICAL_ADAGE_PAIRS: list[tuple[str, str]] = [
    ("look before you leap", "he who hesitates is lost"),
    ("many hands make light work", "too many cooks spoil the broth"),
    ("absence makes the heart grow fonder", "out of sight, out of mind"),
    ("the early bird gets the worm", "fools rush in where angels fear to tread"),
    (
        "you can't judge a book by its cover",
        "clothes make the man",
    ),
    ("two heads are better than one", "if you want something done right, do it yourself"),
    ("haste makes waste", "strike while the iron is hot"),
    ("birds of a feather flock together", "opposites attract"),
    ("actions speak louder than words", "the pen is mightier than the sword"),
    ("great minds think alike", "fools seldom differ"),
    ("better safe than sorry", "nothing ventured, nothing gained"),
    ("the squeaky wheel gets the grease", "silence is golden"),
    ("a penny saved is a penny earned", "you can't take it with you"),
    ("clothes make the man", "don't judge a book by its cover"),
    ("knowledge is power", "ignorance is bliss"),
]


def _normalize(adage: str) -> str:
    """Lowercase, trim, strip trailing punctuation for matching."""
    text = adage.lower().strip()
    text = re.sub(r"[\.\!\?\"',]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class CounterAdageStore:
    """A store of contradicting adage pairs.

    Each pair is symmetric: looking up A returns B, and looking up B
    returns A. Callers extend the store via `add` for domain-specific
    adages (engineering aphorisms, finance heuristics, management
    folk wisdom).
    """

    pairs: list[tuple[str, str]] = field(
        default_factory=lambda: list(CANONICAL_ADAGE_PAIRS)
    )

    def _index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for a, b in self.pairs:
            index[_normalize(a)] = b
            index[_normalize(b)] = a
        return index

    def opposite_of(self, adage: str) -> str | None:
        """Return the contradicting adage, or None if no match.

        Matching is exact after normalization. Use `find_partial` for
        substring matching against natural-language reasoning that
        invokes an adage as part of a longer sentence.
        """
        return self._index().get(_normalize(adage))

    def find_partial(self, text: str) -> list[tuple[str, str]]:
        """Find adages embedded in a longer text and return (adage, opposite) pairs.

        Useful for scanning agent reasoning: if the agent's output
        contains "look before you leap" inside a longer argument, the
        store flags it with its opposite for the agent to reconcile.
        """
        normalized_text = _normalize(text)
        hits: list[tuple[str, str]] = []
        seen: set[str] = set()
        for a, b in self.pairs:
            for adage, opposite in ((a, b), (b, a)):
                key = _normalize(adage)
                if key in seen:
                    continue
                if key in normalized_text:
                    hits.append((adage, opposite))
                    seen.add(key)
        return hits

    def add(self, adage: str, opposite: str) -> None:
        """Extend the store with a new adage pair."""
        if not adage.strip() or not opposite.strip():
            raise ValueError("Adage and opposite must both be non-empty.")
        self.pairs.append((adage, opposite))

    def __len__(self) -> int:
        return len(self.pairs)
