"""LLM integration via the Claude Code CLI.

The Mind's structure (Conditional, Distinction, Question, ...) is already
in place. This module fills that structure with content by calling
`claude -p` with a system prompt that enforces the Langer primitives at
the output layer: conditional language, three-or-more alternatives,
framework transparency, structural JSON-schema enforcement on the
response.

Using the Claude Code CLI rather than the Anthropic SDK directly lets
the harness run on a user's Claude subscription without a separate API
key. The CLI is invoked with `--tools ""` to disable tool use and
`--system-prompt` to replace the default Claude Code system prompt with
the harness's own.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING, Any

from mindful_harness.mind import FirehoseItem, Mind
from mindful_harness.primitives import Conditional, Distinction

if TYPE_CHECKING:
    from mindful_harness.hands import HandResult

CLAUDE_BINARY = os.environ.get("CLAUDE_BINARY", "claude")
DEFAULT_INGESTION_MODEL = os.environ.get("MINDFUL_HARNESS_INGESTION_MODEL", "haiku")
DEFAULT_HANDS_MODEL = os.environ.get("MINDFUL_HARNESS_HANDS_MODEL", "sonnet")
DEFAULT_TIMEOUT_SECONDS = float(
    os.environ.get("MINDFUL_HARNESS_TIMEOUT", "120")
)


SYSTEM_PROMPT = """You are operating inside the Mindful Harness, a substrate for AI-native work inspired by Ellen Langer's research on mindfulness.

Your job: take a single firehose item and propose mindful state updates for a Mind that maintains seven epistemic categories: beliefs, knowledge, questions, interests, curiosities, opportunities, ideas.

Apply Langer's primitives at the output layer:

1. CONDITIONAL LANGUAGE. Say "could be" not "is." Use forward-looking provisional labels ("has been," "so far") instead of absolutes ("is," "remains"). Avoid the word "is" wherever it would force closure.

2. THREE OR MORE ALTERNATIVES. Every belief or knowledge claim carries at least three plausible framings (the chosen one plus two or more alternatives). Two collapses to a binary; three suggests four, five, six.

3. ACTIVE DISTINCTION-MAKING. Note what is novel about this item relative to prior items in the same category. Empty noticing is passive attention, not mindfulness.

4. FRAMEWORK TRANSPARENCY. State the framework you are operating under. State what you are NOT looking for.

5. CERTAINTY AS ALARM. If you find yourself producing a high-confidence claim, flag it for interrogation rather than treating it as load-bearing.

6. COUNTERFACTUAL POSTURE. For any explanation you propose, generate at least one alternative explanation that fits the same evidence.

Empty lists are honest; padding is mindless. If a category has nothing to add, return an empty list rather than inventing content."""


DISTILLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "framework": {
            "type": "string",
            "description": "The lens you are applying to this item.",
        },
        "not_looking_for": {
            "type": "string",
            "description": "What you are explicitly not attending to.",
        },
        "distinctions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "compared_to": {"type": "string"},
                    "noticed": {"type": "string"},
                    "forces_new_category": {"type": "boolean"},
                },
                "required": ["compared_to", "noticed"],
            },
        },
        "belief_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "alternatives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                },
                "required": ["key", "value", "confidence", "alternatives"],
            },
        },
        "knowledge_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "alternatives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                },
                "required": ["key", "value", "confidence", "alternatives"],
            },
        },
        "questions": {"type": "array", "items": {"type": "string"}},
        "interests": {"type": "array", "items": {"type": "string"}},
        "curiosities": {"type": "array", "items": {"type": "string"}},
        "opportunities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "enables": {"type": "string"},
                },
                "required": ["text", "enables"],
            },
        },
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "connects": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text"],
            },
        },
        "self_check": {
            "type": "string",
            "description": "One sentence on what could be wrong about this analysis.",
        },
    },
    "required": ["framework", "not_looking_for"],
}


def _trim_for_context(text: str, limit: int = 1500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " ... [truncated]"


def _summarize_mind(mind: Mind, item_limit: int = 5) -> str:
    parts: list[str] = []
    if mind.beliefs:
        parts.append("Recent beliefs:")
        for key, c in list(mind.beliefs.items())[-item_limit:]:
            parts.append(f"  - {key}: could be {c.value!r} (conf {c.confidence:.2f})")
    if mind.knowledge:
        parts.append("Recent knowledge:")
        for key, c in list(mind.knowledge.items())[-item_limit:]:
            parts.append(f"  - {key}: could be {c.value!r} (conf {c.confidence:.2f})")
    if mind.questions:
        open_qs = [q for q in mind.questions if not q.resolved][-item_limit:]
        if open_qs:
            parts.append("Open questions:")
            parts.extend(f"  - {q.text}" for q in open_qs)
    if mind.interests:
        parts.append("Active interests:")
        parts.extend(f"  - {i.text}" for i in mind.interests[-item_limit:])
    if not parts:
        return "(Mind is empty.)"
    return "\n".join(parts)


def _user_prompt(item: FirehoseItem, mind_summary: str) -> str:
    return f"""Firehose item:
- source: {item.source}
- timestamp: {item.timestamp}
- content: {_trim_for_context(str(item.content))}

Current Mind state:
{mind_summary}

Propose mindful state updates as structured output conforming to the schema."""


class ClaudeCLIError(RuntimeError):
    """Raised when the claude CLI invocation fails."""


def _run_claude(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = DEFAULT_INGESTION_MODEL,
    schema: dict[str, Any] | None = DISTILLATION_SCHEMA,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_budget_usd: float | None = 0.50,
) -> dict[str, Any]:
    """Invoke `claude -p` with our system prompt and structured output.

    The user prompt (which may contain sensitive firehose data such as
    email bodies, dashboard values, or document fragments) is passed
    via stdin rather than argv to avoid exposure through process
    listings on shared hosts.

    Returns the parsed CLI response (the JSON metadata envelope, with
    `structured_output` populated when a schema was provided).
    """
    args: list[str] = [
        CLAUDE_BINARY,
        "-p",
        "--system-prompt",
        system_prompt,
        "--tools",
        "",
        "--model",
        model,
        "--no-session-persistence",
        "--output-format",
        "json",
    ]
    if schema is not None:
        args.extend(["--json-schema", json.dumps(schema)])
    if max_budget_usd is not None:
        args.extend(["--max-budget-usd", str(max_budget_usd)])

    try:
        completed = subprocess.run(
            args,
            input=user_prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise ClaudeCLIError(f"claude -p timed out after {timeout}s") from e

    debug = os.environ.get("MINDFUL_HARNESS_DEBUG")

    if completed.returncode != 0:
        msg = f"claude -p exited {completed.returncode}"
        if debug:
            msg += f": stderr={completed.stderr.strip()[:500]}"
        raise ClaudeCLIError(msg)

    try:
        envelope = json.loads(completed.stdout)
    except json.JSONDecodeError as e:
        msg = "claude -p output is not valid JSON"
        if debug:
            msg += f": {completed.stdout[:500]}"
        raise ClaudeCLIError(msg) from e

    if envelope.get("is_error"):
        msg = "claude reported error"
        if debug:
            msg += f": {envelope}"
        raise ClaudeCLIError(msg)

    return envelope


def distill_item(
    item: FirehoseItem,
    mind: Mind,
    model: str = DEFAULT_INGESTION_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run a single item through the Langer-primitive prompt.

    Returns the distillation dict matching DISTILLATION_SCHEMA, with
    extra `_model`, `_cost_usd`, `_duration_ms` fields appended for
    observability. Callers can pass the result to `apply_distillation`
    to mutate the Mind, or inspect it first.
    """
    user_prompt = _user_prompt(item, _summarize_mind(mind))
    envelope = _run_claude(
        user_prompt=user_prompt,
        model=model,
        schema=DISTILLATION_SCHEMA,
        timeout=timeout,
    )

    structured = envelope.get("structured_output")
    if structured is None:
        # Fallback: try the `result` text field as JSON.
        raw = envelope.get("result", "")
        if raw:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            try:
                structured = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ClaudeCLIError(
                    f"could not parse fallback `result` as JSON: {raw[:500]}"
                ) from e
    if structured is None:
        raise ClaudeCLIError(f"no structured_output in envelope: {envelope}")

    structured["_model"] = model
    structured["_cost_usd"] = envelope.get("total_cost_usd", 0.0)
    structured["_duration_ms"] = envelope.get("duration_ms", 0)
    return structured


def apply_distillation(
    distillation: dict[str, Any], mind: Mind, item: FirehoseItem
) -> None:
    """Mutate the Mind with the structured updates from `distill_item`.

    Done in two phases: first the typed primitives (Conditional,
    Distinction, etc.) are constructed in a build phase that can raise
    ValueError. Only if the build phase succeeds does the commit phase
    mutate the Mind. This prevents partially-applied state when the
    model produces a malformed payload, which in turn prevents the
    caller from double-ingesting the same item on retry.
    """
    framework = distillation.get("framework", "")

    # ---- Build phase: construct typed primitives. May raise. ----

    distinctions_to_add: list[Distinction] = []
    for d in distillation.get("distinctions", []):
        noticed = (d.get("noticed") or "").strip()
        if not noticed:
            continue  # schema should prevent, but stay defensive
        distinctions_to_add.append(
            Distinction(
                item=item.content,
                compared_to=d.get("compared_to", "(prior items)"),
                noticed=noticed,
                forces_new_category=bool(d.get("forces_new_category", False)),
            )
        )

    beliefs_to_add: list[tuple[str, Conditional[Any]]] = []
    for update in distillation.get("belief_updates", []):
        key = (update.get("key") or "").strip()
        if not key:
            continue
        alternatives = list(update.get("alternatives", []))
        if len(alternatives) < 2:
            continue
        beliefs_to_add.append(
            (
                key,
                Conditional(
                    value=update.get("value"),
                    confidence=float(update.get("confidence", 0.5)),
                    alternatives=alternatives,
                    framing=framework,
                ),
            )
        )

    knowledge_to_add: list[tuple[str, Conditional[Any]]] = []
    for update in distillation.get("knowledge_updates", []):
        key = (update.get("key") or "").strip()
        if not key:
            continue
        alternatives = list(update.get("alternatives", []))
        if len(alternatives) < 2:
            continue
        knowledge_to_add.append(
            (
                key,
                Conditional(
                    value=update.get("value"),
                    confidence=float(update.get("confidence", 0.5)),
                    alternatives=alternatives,
                    framing=framework,
                ),
            )
        )

    questions = [q for q in distillation.get("questions", []) if q]
    interests = [i for i in distillation.get("interests", []) if i]
    curiosities = [c for c in distillation.get("curiosities", []) if c]
    opportunities = [
        (o.get("text", ""), o.get("enables", ""))
        for o in distillation.get("opportunities", [])
        if o.get("text")
    ]
    ideas = [
        (idea.get("text", ""), list(idea.get("connects", []) or []))
        for idea in distillation.get("ideas", [])
        if idea.get("text")
    ]

    # ---- Commit phase: only after the build phase succeeded. ----

    mind.ingest(item)
    for d in distinctions_to_add:
        mind.record_distinction(d)
    for key, c in beliefs_to_add:
        mind.believe(key, c)
    for key, c in knowledge_to_add:
        mind.know(key, c)
    for q in questions:
        mind.ask(q)
    for i in interests:
        mind.notice(i)
    for c in curiosities:
        mind.wonder(c)
    for text, enables in opportunities:
        mind.see_opportunity(text=text, enables=enables)
    for text, connects in ideas:
        mind.connect(text=text, connects=connects)


def ingest(
    item: FirehoseItem,
    mind: Mind,
    model: str = DEFAULT_INGESTION_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Convenience entry point: distill an item and apply the result to the Mind.

    Returns the distillation for inspection.
    """
    distillation = distill_item(item, mind, model=model, timeout=timeout)
    apply_distillation(distillation, mind, item)
    return distillation


# ---------------------------------------------------------------------------
# Hand execution via LLM
# ---------------------------------------------------------------------------

HAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chosen": {
            "type": "string",
            "description": "The action chosen, in conditional language.",
        },
        "rejected_alternatives": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "description": "At least two alternatives considered and not chosen.",
        },
        "framings": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "minProperties": 3,
            "description": (
                "At least three framings of the chosen action; the receiver "
                "picks which framing to act under (chambermaid effect)."
            ),
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {
            "type": "string",
            "description": "How the choice was reached, including what was rejected and why.",
        },
        "would_revise_if": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Reversion triggers; conditions that would reverse the choice.",
        },
    },
    "required": ["chosen", "rejected_alternatives", "framings", "confidence", "reasoning"],
}


HAND_SYSTEM_PROMPT = """You are a Hand in the Mindful Harness, spawned by a Mind to take action on a task.

Apply Langer's primitives at the per-action level:

1. CHOOSE WITH ALTERNATIVES. Articulate at least two rejected alternatives alongside your chosen action. A choice without alternatives is a default; defaults are mindless.

2. MULTIPLE FRAMINGS. Return three or more framings of the chosen action, each describing what the action means under a different lens. The receiver of your result picks which framing to act under, which is the chambermaid effect operationalized.

3. CONDITIONAL LANGUAGE. The chosen action is phrased as "could" or "may" rather than "will" or "must."

4. SENSITIVITY TO CONTEXT. The chosen action accounts for who is affected by it and the specific situation, rather than applying a general rule mechanically. Mindfulness here is openness to the particulars of this case.

5. REVERSION TRIGGERS. State what would make you choose differently. A decision without reversion triggers ossifies into invisible commitment.

6. REASONING TRANSPARENCY. Show how the choice was reached, including what was rejected and why.

Return JSON matching the schema. Empty fields are honest; padding is mindless."""


def _hand_user_prompt(hand_task: str, hand_framework: str, snapshot: dict[str, Any]) -> str:
    snapshot_text = json.dumps(snapshot, indent=2, default=str)
    snapshot_text = _trim_for_context(snapshot_text, limit=3000)
    return f"""Task: {hand_task}

Framework: {hand_framework}

Mind state at spawn:
{snapshot_text}

Decide on the action. Return JSON conforming to the schema."""


def execute_hand_llm(
    hand,
    model: str = DEFAULT_HANDS_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> HandResult:
    """Execute a Hand via the Claude CLI.

    Imports HandResult lazily so this module remains optional for users
    who only want the Mind layer. The returned HandResult is fully typed
    and passes the same structural checks as `Hand.execute_manual`.
    """
    from mindful_harness.hands import HandResult

    user_prompt = _hand_user_prompt(
        hand_task=hand.task,
        hand_framework=hand.framework,
        snapshot=hand.mind_snapshot,
    )

    envelope = _run_claude(
        user_prompt=user_prompt,
        system_prompt=HAND_SYSTEM_PROMPT,
        model=model,
        schema=HAND_SCHEMA,
        timeout=timeout,
    )

    structured = envelope.get("structured_output")
    if structured is None:
        raw = envelope.get("result", "").strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        structured = json.loads(raw)

    hand.log(f"LLM execution complete via {model}")

    return HandResult(
        chosen=structured["chosen"],
        rejected_alternatives=list(structured["rejected_alternatives"]),
        framings=dict(structured["framings"]),
        framework=hand.framework,
        confidence=float(structured["confidence"]),
        process_trail=list(hand.process_trail) + [f"reasoning: {structured.get('reasoning', '')[:200]}"],
        would_revise_if=list(structured.get("would_revise_if", [])),
    )
