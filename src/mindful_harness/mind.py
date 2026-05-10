"""The Mind: continuous epistemic state.

The Mind is always on. It ingests a firehose of information and maintains
seven structured state categories (beliefs, knowledge, questions,
interests, curiosities, opportunities, ideas), each updated through the
Langer primitives as new items arrive.

This is the v0.0.x scaffolding: state stores, ingestion entry point,
query API. LLM-driven distinction-making and category creation will land
in subsequent revisions; for now ingestion stores items as Conditionals
and the Mind exposes a queryable view.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from mindful_harness.primitives import Conditional, Decision, Distinction, is_certain


@dataclass
class FirehoseItem:
    """A single item arriving from the firehose.

    Items can be email, document chunks, calendar events, news, agent
    outputs, user messages, structured data updates. The Mind treats
    them uniformly at ingestion time and lets distinction-making
    determine where each item belongs.
    """

    source: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    """An open inquiry the Mind is holding.

    Questions are first-class state, not afterthoughts. They are routed:
    to the firehose (look for answers), to Hands (investigate), or to a
    human (ask). A question without a routing target is a question the
    Mind has noticed but not yet acted on.
    """

    text: str
    born_at: float = field(default_factory=time.time)
    routed_to: str | None = None  # "firehose" | "hands" | "human" | None
    resolved: bool = False
    resolution: str | None = None


@dataclass
class Interest:
    """An anomaly or deviation the Mind has noticed.

    Interests decay over time unless reinforced by further evidence.
    The Mind's interest list is a moving window of what is currently
    worth attention.
    """

    text: str
    born_at: float = field(default_factory=time.time)
    last_reinforced: float = field(default_factory=time.time)
    intensity: float = 1.0  # decays with neglect

    def reinforce(self) -> None:
        self.last_reinforced = time.time()
        self.intensity = min(1.0, self.intensity + 0.2)

    def decay(self, half_life_seconds: float = 86400.0) -> None:
        elapsed = time.time() - self.last_reinforced
        if elapsed > 0 and half_life_seconds > 0:
            self.intensity *= 0.5 ** (elapsed / half_life_seconds)


@dataclass
class Curiosity:
    """A directed thread the Mind wants to follow.

    Different from an Interest in being intent-driven rather than
    anomaly-driven. The Mind wants to know more about X.
    """

    text: str
    born_at: float = field(default_factory=time.time)
    pursued: bool = False


@dataclass
class Opportunity:
    """An action-implication of new information.

    What does this enable that wasn't possible before? Opportunities
    are framed as possibilities the Mind has noticed: what new action
    space has opened, for whom, under what framing.
    """

    text: str
    enables: str
    born_at: float = field(default_factory=time.time)
    acted_on: bool = False


@dataclass
class Idea:
    """A creative connection across domains.

    Ideas are bisociations: insights that link items from distant
    categories (the chambermaid effect transferred to debugging labels;
    Halley's-comet inversion transferred to product framing).
    """

    text: str
    connects: list[str]  # the categories or domains being linked
    born_at: float = field(default_factory=time.time)


@dataclass
class Mind:
    """The continuous epistemic state.

    Seven categories of structured state plus an ingestion log. The Mind
    is queried, not commanded; downstream agents and humans ask it what
    it currently believes, what it is curious about, what opportunities
    it has noticed. The Mind also pushes notifications when state
    crosses thresholds.
    """

    beliefs: dict[str, Conditional[Any]] = field(default_factory=dict)
    knowledge: dict[str, Conditional[Any]] = field(default_factory=dict)
    questions: list[Question] = field(default_factory=list)
    interests: list[Interest] = field(default_factory=list)
    curiosities: list[Curiosity] = field(default_factory=list)
    opportunities: list[Opportunity] = field(default_factory=list)
    ideas: list[Idea] = field(default_factory=list)
    decisions: list[Decision[Any]] = field(default_factory=list)

    ingestion_log: list[FirehoseItem] = field(default_factory=list)
    distinction_log: list[Distinction] = field(default_factory=list)

    action_history: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """Tracks repeated actions for the habituation alarm.

    Maps action signature to a list of timestamps when it was executed.
    """

    def ingest(self, item: FirehoseItem) -> None:
        """Take in a firehose item.

        v0.0.x: stores the item in the ingestion log. Real distinction-
        making, category routing, and state updates land when LLM
        integration arrives. The hook is here so the rest of the Mind
        can be exercised end-to-end without it.
        """
        self.ingestion_log.append(item)

    def believe(self, key: str, value: Conditional[Any]) -> None:
        """Set or update a belief.

        Beliefs are conditional and carry alternatives. Setting an
        absolute belief (no alternatives) is structurally impossible
        because Conditional rejects fewer than three framings.
        """
        self.beliefs[key] = value

    def know(self, key: str, value: Conditional[Any]) -> None:
        """Record a factual claim with provenance and alternatives.

        Same shape as belief; semantically distinguished — beliefs are
        the Mind's working model, knowledge is what the Mind has
        evidence for.
        """
        self.knowledge[key] = value

    def ask(self, text: str, routed_to: str | None = None) -> Question:
        """Open a question. Returns the Question for further routing."""
        q = Question(text=text, routed_to=routed_to)
        self.questions.append(q)
        return q

    def notice(self, text: str) -> Interest:
        """Note an interest (anomaly, deviation, novelty). Returns the Interest."""
        for existing in self.interests:
            if existing.text == text:
                existing.reinforce()
                return existing
        interest = Interest(text=text)
        self.interests.append(interest)
        return interest

    def wonder(self, text: str) -> Curiosity:
        """Note a curiosity (intent-driven thread). Returns the Curiosity."""
        c = Curiosity(text=text)
        self.curiosities.append(c)
        return c

    def see_opportunity(self, text: str, enables: str) -> Opportunity:
        """Note an opportunity. Returns the Opportunity."""
        o = Opportunity(text=text, enables=enables)
        self.opportunities.append(o)
        return o

    def connect(self, text: str, connects: list[str]) -> Idea:
        """Note a creative connection across domains. Returns the Idea."""
        i = Idea(text=text, connects=connects)
        self.ideas.append(i)
        return i

    def commit(self, decision: Decision[Any]) -> None:
        """Record a decision (held loosely, with reversion triggers)."""
        self.decisions.append(decision)

    def record_distinction(self, d: Distinction) -> None:
        """Log an explicit distinction noted during ingestion or reasoning."""
        self.distinction_log.append(d)

    def record_action(self, signature: str) -> int:
        """Record that an action with the given signature was executed.

        Returns the count of identical executions, including this one.
        Counts at or above 3 trip the habituation alarm.
        """
        self.action_history[signature].append(time.time())
        return len(self.action_history[signature])

    def is_habituated(self, signature: str, threshold: int = 3) -> bool:
        """Has the system executed this action pattern enough times to alarm?

        Per the v0.5 success-as-scrutiny-trigger inversion: a True
        return is a signal to investigate, not a signal that the
        action is safe.
        """
        return len(self.action_history.get(signature, [])) >= threshold

    def certainty_alarms(self) -> list[tuple[str, Conditional[Any]]]:
        """Return beliefs and knowledge that have crossed the certainty threshold.

        Per Langer: certainty is the signal of mindlessness. The Mind
        surfaces these for interrogation rather than treating them as
        load-bearing.
        """
        alarms: list[tuple[str, Conditional[Any]]] = []
        for key, c in self.beliefs.items():
            if is_certain(c):
                alarms.append((f"belief:{key}", c))
        for key, c in self.knowledge.items():
            if is_certain(c):
                alarms.append((f"knowledge:{key}", c))
        return alarms

    def vital_signs(self) -> dict[str, float]:
        """The Mind's behavioral markers of mindfulness.

        Drawn from Langer's three operational characteristics
        (Mindfulness, 1989, p. 41-62): category creation, openness to
        new information, awareness of multiple perspectives. These are
        the harness's vital signs, separate from any task metric.
        """
        total_alternatives = sum(
            len(c.alternatives) for c in self.beliefs.values()
        ) + sum(len(c.alternatives) for c in self.knowledge.values())
        total_conditionals = len(self.beliefs) + len(self.knowledge)
        avg_alternatives = (
            total_alternatives / total_conditionals if total_conditionals else 0.0
        )
        return {
            "alternative_cardinality": avg_alternatives,
            "open_questions": float(
                sum(1 for q in self.questions if not q.resolved)
            ),
            "active_interests": float(len(self.interests)),
            "curiosities_pursued": float(
                sum(1 for c in self.curiosities if c.pursued)
            ),
            "opportunities_pending": float(
                sum(1 for o in self.opportunities if not o.acted_on)
            ),
            "decisions_held": float(len(self.decisions)),
            "distinctions_noted": float(len(self.distinction_log)),
            "items_ingested": float(len(self.ingestion_log)),
        }
