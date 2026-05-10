"""End-to-end demo: a Mind ingesting a small firehose with LLM distillation.

Run from the harness root directory:

    PYTHONPATH=src python3 examples/founders_inbox.py

This exercises the full ingestion pipeline: each mock firehose item is run
through the Langer-primitive prompt, the structured response is applied to
the Mind via the typed primitives, and the resulting state is printed.

Costs roughly $0.01-0.05 per run depending on item count and model.
"""

from __future__ import annotations

import sys
import time

from mindful_harness import Mind
from mindful_harness.llm import ingest
from mindful_harness.mind import FirehoseItem

MOCK_FIREHOSE: list[FirehoseItem] = [
    FirehoseItem(
        source="email",
        content=(
            "From: partnerships@vendor-x.com\n"
            "Subject: New API for direct integration\n\n"
            "Hi team. We're rolling out a new public API next month that lets "
            "your platform integrate with ours directly, without the webhook "
            "workaround your engineers built last year. Beta access available now."
        ),
    ),
    FirehoseItem(
        source="dashboard",
        content=(
            "Q1 metrics update: Revenue +12% YoY, customer support ticket volume "
            "-23%, NPS up from 41 to 53. No marketing campaign changes in the period."
        ),
    ),
    FirehoseItem(
        source="news",
        content=(
            "Competitor Acme Co. announced a Series C raise at $1.2B valuation, "
            "down round from their last raise. They cited 'market headwinds' "
            "and laid off 18% of staff."
        ),
    ),
    FirehoseItem(
        source="calendar",
        content=(
            "Tomorrow 9am: 1:1 with engineering lead. They flagged in chat "
            "that the migration is 'mostly done' and they want to discuss timeline."
        ),
    ),
    FirehoseItem(
        source="customer-success",
        content=(
            "Three enterprise customers in the same week asked unprompted "
            "about international expansion. None had asked before."
        ),
    ),
]


def print_separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_distillation(distillation: dict, item: FirehoseItem) -> None:
    print(f"Item from {item.source}:")
    print(f"  framework: {distillation.get('framework', '')}")
    print(f"  not looking for: {distillation.get('not_looking_for', '')}")

    distinctions = distillation.get("distinctions", [])
    if distinctions:
        print(f"  distinctions ({len(distinctions)}):")
        for d in distinctions:
            print(f"    - {d.get('noticed', '')}")

    beliefs = distillation.get("belief_updates", [])
    if beliefs:
        print(f"  belief updates ({len(beliefs)}):")
        for b in beliefs:
            alts = b.get("alternatives", [])
            print(
                f"    - {b.get('key', '')}: could be {b.get('value', '')!r} "
                f"(conf {b.get('confidence', 0):.2f}, {len(alts)} alternatives)"
            )

    questions = distillation.get("questions", [])
    if questions:
        print(f"  questions ({len(questions)}):")
        for q in questions:
            print(f"    - {q}")

    interests = distillation.get("interests", [])
    if interests:
        print(f"  interests ({len(interests)}):")
        for i in interests:
            print(f"    - {i}")

    opportunities = distillation.get("opportunities", [])
    if opportunities:
        print(f"  opportunities ({len(opportunities)}):")
        for o in opportunities:
            print(f"    - {o.get('text', '')}")
            print(f"      enables: {o.get('enables', '')}")

    ideas = distillation.get("ideas", [])
    if ideas:
        print(f"  ideas ({len(ideas)}):")
        for idea in ideas:
            print(f"    - {idea.get('text', '')}")

    self_check = distillation.get("self_check", "")
    if self_check:
        print(f"  self-check: {self_check}")
    print()


def print_mind_summary(mind: Mind) -> None:
    print("Beliefs:")
    for key, c in mind.beliefs.items():
        print(f"  - {key}: could be {c.value!r} (conf {c.confidence:.2f}, {len(c.alternatives)} alts)")

    print("\nKnowledge:")
    for key, c in mind.knowledge.items():
        print(f"  - {key}: could be {c.value!r} (conf {c.confidence:.2f}, {len(c.alternatives)} alts)")

    print(f"\nOpen questions ({sum(1 for q in mind.questions if not q.resolved)}):")
    for q in mind.questions:
        if not q.resolved:
            print(f"  - {q.text}")

    print(f"\nInterests ({len(mind.interests)}):")
    for i in mind.interests:
        print(f"  - {i.text}")

    print(f"\nOpportunities ({len(mind.opportunities)}):")
    for o in mind.opportunities:
        print(f"  - {o.text}")
        print(f"    enables: {o.enables}")

    print(f"\nIdeas ({len(mind.ideas)}):")
    for idea in mind.ideas:
        print(f"  - {idea.text}")
        print(f"    connects: {idea.connects}")

    alarms = mind.certainty_alarms()
    if alarms:
        print(f"\nCertainty alarms ({len(alarms)}):")
        for name, c in alarms:
            print(f"  - {name}: confidence {c.confidence:.2f} — interrogate")

    print("\nVital signs:")
    for name, value in mind.vital_signs().items():
        print(f"  - {name}: {value:.2f}")


def main() -> int:
    mind = Mind()

    print_separator("INGESTING FIREHOSE")
    start = time.time()
    for i, item in enumerate(MOCK_FIREHOSE, 1):
        print(f"\n[{i}/{len(MOCK_FIREHOSE)}] Distilling item from {item.source}...")
        try:
            distillation = ingest(item, mind)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print_distillation(distillation, item)
    elapsed = time.time() - start
    print(f"Ingestion complete in {elapsed:.1f}s.")

    print_separator("MIND STATE AFTER INGESTION")
    print_mind_summary(mind)

    return 0


if __name__ == "__main__":
    sys.exit(main())
