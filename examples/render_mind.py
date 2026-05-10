"""Render a Mind to HTML for human reading.

Builds a populated Mind without LLM calls (the founders-inbox scenario,
but with manual primitives so the script costs nothing to run) and
writes an HTML snapshot to mind.html in the current directory. Open in
any browser.

    PYTHONPATH=src python3 examples/render_mind.py

The companion `founders_inbox.py` is the live LLM version; this script
exercises the same Mind structures so the rendering can be iterated
without the latency or cost of distillation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mindful_harness import (
    Conditional,
    Decision,
    FirehoseItem,
    Mind,
    render_mind_html,
)


def build_demo_mind() -> Mind:
    m = Mind()

    m.ingest(FirehoseItem(source="email", content="Vendor X announced an API"))
    m.ingest(FirehoseItem(source="dashboard", content="Q1 metrics"))
    m.ingest(FirehoseItem(source="news", content="Acme competitor news"))

    m.believe(
        "Q1 trend",
        Conditional(
            value="revenue up 12% YoY",
            confidence=0.7,
            alternatives=["flat with measurement noise", "Q4 carryover effect"],
            framing="under the assumption that the dashboard cache is fresh",
            reversion_trigger="if the data refresh failed silently",
        ),
    )
    m.believe(
        "support load anomaly",
        Conditional(
            value="dropped 23% while NPS rose 12 points",
            confidence=0.6,
            alternatives=[
                "could be retention-driven, not satisfaction-driven",
                "could be ticket-classification change masking real volume",
            ],
        ),
    )
    m.know(
        "vendor announced API",
        Conditional(
            value=True,
            confidence=0.95,
            alternatives=[False, "rumor not yet confirmed"],
        ),
    )

    m.ask("What drove the Q1 lift, given no marketing change?", routed_to="hands")
    m.ask("Does the new vendor API actually fit our integration pattern?")
    m.ask("Is the Acme down round a market signal or company-specific?")

    m.notice("support load dropped while NPS rose, same period, no obvious cause")
    m.notice("vendor reached out about a workaround they apparently noticed")
    m.notice("three enterprise customers asked about international expansion, same week")

    m.wonder("how does the Tuesday cohort retention compare to Monday's?")
    m.wonder("what would the chambermaid framing tell us about the support drop?")

    m.see_opportunity(
        text="Share Q1 metrics with vendor as market intelligence rather than negotiation",
        enables="vendor learns integration is competitive feature, preserves their agency to design API well",
    )
    m.see_opportunity(
        text="Parallel test new vendor API in staging alongside existing webhook",
        enables="empirical risk assessment instead of vendor-framed urgency",
    )
    m.see_opportunity(
        text="Approach Acme for partnership not poaching",
        enables="ecosystem partnership that preserves their agency under funding pressure",
    )

    m.connect(
        text="the support drop without marketing change resembles the chambermaid effect",
        connects=["customer-support", "chambermaid-study", "framing-rewrites-physiology"],
    )
    m.connect(
        text="vendor 'observed your workaround' could be platform surveillance OR healthy responsiveness, both plausible",
        connects=["vendor-relations", "trust-and-gullibility"],
    )

    m.commit(
        Decision(
            chosen="investigate Q1 driver before acting",
            rejected=["assume seasonal effect", "assume measurement noise", "assume marketing lag"],
            framing="initial Q1 review",
            reversion_triggers=[
                "if dashboard cache proves stale",
                "if Q2 reverses the trend",
                "if cohort analysis shows composition change",
            ],
            confidence=0.65,
        )
    )

    return m


def main() -> int:
    mind = build_demo_mind()
    html = render_mind_html(mind, title="Founder Inbox, Mid-Q1")

    output_path = Path("mind.html")
    output_path.write_text(html)

    print(f"Mind state rendered to {output_path.resolve()}")
    print(f"  beliefs: {len(mind.beliefs)}")
    print(f"  knowledge: {len(mind.knowledge)}")
    print(f"  open questions: {sum(1 for q in mind.questions if not q.resolved)}")
    print(f"  interests: {len(mind.interests)}")
    print(f"  opportunities: {len(mind.opportunities)}")
    print(f"  ideas: {len(mind.ideas)}")
    print(f"  decisions: {len(mind.decisions)}")
    print(f"  certainty alarms: {len(mind.certainty_alarms())}")
    print()
    print(f"Open in a browser: file://{output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
