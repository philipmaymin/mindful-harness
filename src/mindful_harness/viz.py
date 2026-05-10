"""Static HTML rendering of Mind state.

Given a Mind, render its seven epistemic categories as a single HTML
page that a human can read. No server, no polling — just a snapshot,
because the Mind's state is meant to be looked at, not animated.

A separate web server adapter could mount this output and refresh it
on ingestion, but the rendering itself is pure: Mind -> HTML string.
"""

from __future__ import annotations

import html
import json
from datetime import datetime

from mindful_harness.mind import Mind

_CSS = """
:root {
  --bg: #fafaf7;
  --ink: #2a2826;
  --muted: #6b6660;
  --accent: #4a6741;
  --alarm: #a64242;
  --rule: #d6d0c4;
  --card: #ffffff;
  --shadow: 0 1px 3px rgba(40, 36, 30, 0.06);
}
* { box-sizing: border-box; }
body {
  font: 15px/1.55 'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif;
  background: var(--bg);
  color: var(--ink);
  margin: 0;
  padding: 40px 24px 80px;
}
.wrap { max-width: 880px; margin: 0 auto; }
header {
  border-bottom: 1px solid var(--rule);
  padding-bottom: 16px;
  margin-bottom: 32px;
}
h1 {
  font-size: 28px;
  font-weight: 600;
  margin: 0 0 4px;
  letter-spacing: -0.01em;
}
.subtitle {
  color: var(--muted);
  font-size: 14px;
}
section { margin-bottom: 36px; }
h2 {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 600;
  color: var(--muted);
  border-bottom: 1px solid var(--rule);
  padding-bottom: 6px;
  margin: 0 0 14px;
}
.item {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 14px 16px;
  margin-bottom: 10px;
  box-shadow: var(--shadow);
}
.item .key {
  font-weight: 600;
  color: var(--ink);
  font-size: 14px;
}
.item .value {
  margin: 4px 0 8px;
  color: var(--ink);
}
.item .alts {
  color: var(--muted);
  font-size: 13px;
  font-style: italic;
}
.item .conf {
  display: inline-block;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  color: var(--muted);
  letter-spacing: 0.04em;
  padding: 2px 6px;
  border: 1px solid var(--rule);
  border-radius: 3px;
}
.item .conf.alarm {
  color: var(--alarm);
  border-color: var(--alarm);
}
.simple-list { list-style: none; padding: 0; margin: 0; }
.simple-list li {
  padding: 6px 0;
  border-bottom: 1px dashed var(--rule);
}
.simple-list li:last-child { border-bottom: none; }
.opportunity { padding: 10px 14px; }
.opportunity .enables {
  color: var(--accent);
  font-size: 13px;
  margin-top: 4px;
  padding-left: 12px;
  border-left: 2px solid var(--accent);
}
.idea .connects {
  color: var(--muted);
  font-size: 12px;
  font-style: italic;
  margin-top: 4px;
}
.vitals {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.vital {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 10px 12px;
}
.vital .name {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
}
.vital .number {
  font-size: 22px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.alarm-banner {
  background: #fff4f0;
  border: 1px solid #e5c4ba;
  color: var(--alarm);
  padding: 10px 14px;
  border-radius: 4px;
  margin-bottom: 24px;
  font-size: 14px;
}
footer {
  margin-top: 64px;
  padding-top: 16px;
  border-top: 1px solid var(--rule);
  color: var(--muted);
  font-size: 12px;
  text-align: center;
}
.empty {
  color: var(--muted);
  font-style: italic;
  font-size: 13px;
}
"""


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _alts_line(alts: list) -> str:
    if not alts:
        return ""
    items = " / ".join(_esc(a) for a in alts)
    return f'<div class="alts">could also be: {items}</div>'


def _belief_block(key: str, c) -> str:
    conf_pct = f"{c.confidence * 100:.0f}%"
    alarm_class = " alarm" if c.confidence >= 0.9 else ""
    return f"""
      <div class="item">
        <div class="key">{_esc(key)}</div>
        <div class="value">could be {_esc(c.value)}</div>
        {_alts_line(c.alternatives)}
        <span class="conf{alarm_class}">conf {conf_pct}</span>
      </div>
    """.strip()


def _simple_list(items: list[str]) -> str:
    if not items:
        return '<p class="empty">none yet</p>'
    lines = "".join(f"<li>{_esc(t)}</li>" for t in items)
    return f'<ul class="simple-list">{lines}</ul>'


def render_mind_html(mind: Mind, title: str = "Mind") -> str:
    """Render a Mind's current state as a complete HTML page."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    beliefs_html = (
        "\n".join(_belief_block(k, c) for k, c in mind.beliefs.items())
        if mind.beliefs
        else '<p class="empty">no beliefs yet</p>'
    )
    knowledge_html = (
        "\n".join(_belief_block(k, c) for k, c in mind.knowledge.items())
        if mind.knowledge
        else '<p class="empty">no knowledge yet</p>'
    )

    open_questions = [q.text for q in mind.questions if not q.resolved]
    interests = [i.text for i in mind.interests]
    curiosities = [c.text for c in mind.curiosities]

    if mind.opportunities:
        opp_items = "".join(
            f"""
            <div class="item opportunity">
              <div>{_esc(o.text)}</div>
              <div class="enables">enables: {_esc(o.enables)}</div>
            </div>
            """
            for o in mind.opportunities
        )
    else:
        opp_items = '<p class="empty">no opportunities surfaced yet</p>'

    if mind.ideas:
        idea_items = "".join(
            f"""
            <div class="item idea">
              <div>{_esc(i.text)}</div>
              <div class="connects">connects: {_esc(" + ".join(i.connects))}</div>
            </div>
            """
            for i in mind.ideas
        )
    else:
        idea_items = '<p class="empty">no creative connections yet</p>'

    if mind.decisions:
        decision_items = "".join(
            f"""
            <div class="item">
              <div class="key">chose {_esc(d.chosen)}</div>
              <div class="value">over {_esc(d.rejected)}</div>
              <div class="alts">framing: {_esc(d.framing)}</div>
              <div class="alts">revisit if: {_esc("; ".join(d.reversion_triggers))}</div>
            </div>
            """
            for d in mind.decisions
        )
    else:
        decision_items = '<p class="empty">no decisions committed</p>'

    alarms = mind.certainty_alarms()
    if alarms:
        alarm_lines = "".join(
            f"<li>{_esc(name)} (conf {c.confidence * 100:.0f}%) — interrogate</li>"
            for name, c in alarms
        )
        alarm_html = f"""
          <div class="alarm-banner">
            <strong>Certainty alarms.</strong> These items have crossed the threshold
            where confidence is itself the signal of mindlessness. Interrogate before trusting.
            <ul style="margin: 6px 0 0; padding-left: 18px;">{alarm_lines}</ul>
          </div>
        """
    else:
        alarm_html = ""

    vitals = mind.vital_signs()
    vital_blocks = "".join(
        f"""
        <div class="vital">
          <div class="name">{_esc(name.replace("_", " "))}</div>
          <div class="number">{value:.2f}</div>
        </div>
        """
        for name, value in vitals.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{_esc(title)} | Mindful Harness</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>{_esc(title)}</h1>
      <div class="subtitle">A snapshot of a Mind, rendered {now}.</div>
    </header>

    {alarm_html}

    <section>
      <h2>Vital signs</h2>
      <div class="vitals">{vital_blocks}</div>
    </section>

    <section>
      <h2>Beliefs</h2>
      {beliefs_html}
    </section>

    <section>
      <h2>Knowledge</h2>
      {knowledge_html}
    </section>

    <section>
      <h2>Open questions</h2>
      {_simple_list(open_questions)}
    </section>

    <section>
      <h2>Interests</h2>
      {_simple_list(interests)}
    </section>

    <section>
      <h2>Curiosities</h2>
      {_simple_list(curiosities)}
    </section>

    <section>
      <h2>Opportunities</h2>
      {opp_items}
    </section>

    <section>
      <h2>Ideas</h2>
      {idea_items}
    </section>

    <section>
      <h2>Decisions held</h2>
      {decision_items}
    </section>

    <footer>
      Mindful Harness, a substrate inspired by Ellen Langer's mindfulness research,
      coined with Philip Maymin. Outputs are conditional, not absolute.
    </footer>
  </div>
</body>
</html>
"""


def render_mind_json(mind: Mind) -> str:
    """Render Mind state as JSON for programmatic consumption.

    Useful for a web visualizer that polls a separate endpoint, or for
    archival snapshots of Mind state at a particular moment.
    """
    return json.dumps(
        {
            "beliefs": {
                k: {
                    "value": c.value,
                    "confidence": c.confidence,
                    "alternatives": c.alternatives,
                    "framing": c.framing,
                }
                for k, c in mind.beliefs.items()
            },
            "knowledge": {
                k: {
                    "value": c.value,
                    "confidence": c.confidence,
                    "alternatives": c.alternatives,
                    "framing": c.framing,
                }
                for k, c in mind.knowledge.items()
            },
            "questions": [
                {"text": q.text, "routed_to": q.routed_to, "resolved": q.resolved}
                for q in mind.questions
            ],
            "interests": [
                {"text": i.text, "intensity": i.intensity}
                for i in mind.interests
            ],
            "curiosities": [
                {"text": c.text, "pursued": c.pursued} for c in mind.curiosities
            ],
            "opportunities": [
                {"text": o.text, "enables": o.enables, "acted_on": o.acted_on}
                for o in mind.opportunities
            ],
            "ideas": [
                {"text": i.text, "connects": i.connects} for i in mind.ideas
            ],
            "decisions": [
                {
                    "chosen": str(d.chosen),
                    "rejected": [str(r) for r in d.rejected],
                    "framing": d.framing,
                    "reversion_triggers": d.reversion_triggers,
                    "confidence": d.confidence,
                    "age_seconds": d.age_seconds(),
                }
                for d in mind.decisions
            ],
            "vital_signs": mind.vital_signs(),
            "certainty_alarms": [
                {"name": name, "confidence": c.confidence}
                for name, c in mind.certainty_alarms()
            ],
        },
        indent=2,
        default=str,
    )
