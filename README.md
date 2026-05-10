# Mindful Harness

A substrate for AI-native work, inspired by Ellen Langer's research on mindfulness.

## Why

Existing AI harnesses (Claude Code, AutoGen, CrewAI, LangGraph, MetaGPT, ChatDev, OpenAI Agents SDK) optimize for goal completion on bounded tasks. They are amplifiers of group data: confident, fluent, frequently in error but rarely in doubt. They restart from zero on each task.

The Mindful Harness optimizes for **quality of attention sustained over time**. It has a **Mind** that ingests a firehose and maintains an evolving epistemic state. It has **Hands** that act when the Mind notices something action-worthy. Goal completion is one output of attention well-paid; it is not the design objective.

**Mindfulness is good.** Mindlessness is the source of avoidable harm in hospitals, schools, marriages, markets, and now in AI systems trained on group data. A fully mindful AI substrate is not morally neutral; it is the alternative to systems that automate mindlessness at scale. The disposition that earlier drafts attributed to a separate "kindfulness" primitive is now understood as a natural consequence of full mindfulness: sensitivity to context, awareness of multiple perspectives, and openness to new information all imply seeing the counterpart clearly.

## Architecture in one image

```
┌────────────────────────────────────────────────────────────┐
│  Firehose (email, docs, calendar, news, agents, user)      │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │            THE MIND              │
        │  Beliefs, Knowledge, Questions,  │
        │  Interests, Curiosities,         │
        │  Opportunities, Ideas            │
        │                                  │
        │  Drift detector, certainty alarm │
        │  Vital signs, habituation alarm  │
        └──────────────────────────────────┘
            │                │
            │ (query)        │ (spawn when stake-worthy)
            ▼                ▼
        ┌─────────┐     ┌────────────────────┐
        │  User / │     │     THE HANDS      │
        │  Other  │     │  Worker agents     │
        │  Agents │     │  Mindful advocate  │
        └─────────┘     │  Framework         │
                        │  questioner        │
                        └────────────────────┘
                                 │
                                 ▼
                         (results flow back to Mind)
```

## Ten primitives

Each primitive is a structural lock at the type level rather than a prompt instruction. See `SPEC.md` for the full design and the Langer sources for each.

1. **Conditional language and provisional labeling**, outputs use "could be" not "is."
2. **Certainty as alarm**, high-confidence outputs trigger interrogation, not trust.
3. **Three-or-more generation**, forward (alternatives) and backward (counterfactuals).
4. **Counter-adage surfacing**, when folk wisdom contradicts itself, name it.
5. **Look-forward-one-step**, articulate the second action, not just the first.
6. **Drift detection, open categorization, habituation alarm**, successful repetition is a warning sign, not a confirmation.
7. **Framework transparency and questioning**, state assumptions and what is not being looked for; a separate agent challenges the lens.
8. **Decisions held loosely**, every commitment carries its reversion triggers.
9. **Anomaly probing (not escalation)**, investigate before deferring to humans.
10. **Active distinction-making**, what is novel about this item relative to prior items?

## Install

```bash
pip install -e .          # editable install from this directory
# OR, when published:
# pip install mindful-harness
```

LLM features use the [Claude Code](https://claude.com/claude-code) CLI rather than a separate API key, so a Claude subscription is sufficient. Set `CLAUDE_BINARY` if `claude` is not on your PATH.

## Quick start (CLI)

```bash
# Open beliefs, questions, interests, etc.
mindful-harness believe 'Q1 trend' 'revenue up 12%' --alt 'flat with noise' --alt 'Q4 carryover' --confidence 0.7
mindful-harness ask 'what drove Q1 lift?' --routed-to hands
mindful-harness notice 'support load dropped while NPS rose'
mindful-harness wonder 'how does Tuesday cohort differ?'

# Add a firehose item (no LLM)
mindful-harness ingest --source email --content "Vendor X announced an API"

# Add a firehose item with LLM distillation (uses your Claude subscription)
mindful-harness ingest --source email --from-stdin --with-llm < message.txt

# Inspect Mind state
mindful-harness status
mindful-harness vitals
mindful-harness alarms       # outputs that crossed the certainty threshold
mindful-harness drift        # state that has aged past its threshold
mindful-harness show --open  # render to HTML and open in browser
```

Mind state persists at `~/.mindful-harness/mind.json` by default. Override via `--state` or `MINDFUL_HARNESS_STATE`.

## Quick start (Python)

```python
from mindful_harness import Conditional, Mind, FirehoseItem
from mindful_harness.llm import ingest

mind = Mind()

# Manual primitives
mind.believe("Q1 trend", Conditional(
    value="up 12%",
    confidence=0.7,
    alternatives=["flat with noise", "Q4 carryover"],
    framing="under the assumption the dashboard is fresh",
    reversion_trigger="if data refresh failed silently",
))
mind.ask("what drove Q1 lift?", routed_to="hands")
mind.notice("support load dropped while NPS rose")

# LLM-driven ingestion (uses claude -p, no API key required)
item = FirehoseItem(source="email", content="Vendor X announced an API")
distillation = ingest(item, mind)
print(distillation["framework"])
print(distillation["opportunities"])

# Inspect state
print(mind.vital_signs())
for name, c in mind.certainty_alarms():
    print(f"alarm: {name} (conf {c.confidence:.2f})")
```

## Status

v0.0.7, pre-alpha. The architecture is in place; the structural locks are in place; the LLM integration works; the CLI is usable. What is not yet built: real firehose connectors (IMAP, RSS, file watchers), the mindful advocate as a separately-instantiated LLM process, a backend-driven web app.

## Interactive demo

A standalone in-browser demo lives at `docs/index.html` and is published via GitHub Pages at:

**https://philipmaymin.github.io/mindful-harness/**

Add beliefs, questions, observations, and opportunities; watch the seven epistemic categories and vital signs update. Three-or-more enforcement is structural (the form rejects a belief with fewer than two alternatives). The certainty alarm banner fires when a belief crosses confidence 0.9. No backend, no install, no API key.

See `SPEC.md` for the full design and the Langer sources behind each primitive.

## Tests

```bash
PYTHONPATH=src python3 -m pytest tests/
```

147 tests at last count, all structural locks covered. The LLM integration tests mock the CLI invocation; a live end-to-end example lives in `examples/founders_inbox.py`.

## Credits

The thinking is Ellen Langer's, from half a century of research on mindfulness. The project is led by Philip Maymin. The harness is one application of the ideas in their in-progress book MINDLESSNESS.

## License

MIT.
