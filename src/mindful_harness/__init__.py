"""Mindful Harness: a substrate for AI-native work.

The harness is organized in two layers. The Mind ingests a firehose of
information and maintains a continuous epistemic state across seven
categories: beliefs, knowledge, questions, interests, curiosities,
opportunities, and ideas. The Hands are agents spawned when the Mind
notices something action-worthy.

Both layers run on the same primitives, drawn from Ellen Langer's
research on mindfulness. See SPEC.md in the project root for the
full design.
"""

from mindful_harness.hands import (
    Advocate,
    FrameworkQuestioner,
    Hand,
    HandResult,
    run_with_advocate_and_questioner,
    spawn_hand,
)
from mindful_harness.viz import render_mind_html, render_mind_json
from mindful_harness.mind import (
    Curiosity,
    FirehoseItem,
    Idea,
    Interest,
    Mind,
    Opportunity,
    Question,
)
from mindful_harness.primitives import (
    CERTAINTY_THRESHOLD,
    Conditional,
    Decision,
    Distinction,
    is_certain,
)

__version__ = "0.0.7"

__all__ = [
    "Advocate",
    "CERTAINTY_THRESHOLD",
    "Conditional",
    "Curiosity",
    "Decision",
    "Distinction",
    "FirehoseItem",
    "FrameworkQuestioner",
    "Hand",
    "HandResult",
    "Idea",
    "Interest",
    "Mind",
    "Opportunity",
    "Question",
    "is_certain",
    "render_mind_html",
    "render_mind_json",
    "run_with_advocate_and_questioner",
    "spawn_hand",
    "__version__",
]
