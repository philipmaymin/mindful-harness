"""Command-line entry point for the Mindful Harness.

The CLI is the primary way someone exercises the harness without writing
Python. State is persisted to `~/.mindful-harness/mind.json` (overridable
via `--state` or the MINDFUL_HARNESS_STATE env var), so commands compose
across invocations.

Usage examples:

    mindful-harness ingest --source email --content "Vendor X..."
    mindful-harness ingest --source rss --from-stdin
    mindful-harness believe revenue 'up 12%' --alt 'flat' --alt 'down' --confidence 0.7
    mindful-harness ask "what drove Q1 lift?"
    mindful-harness notice "support load dropped while NPS rose"
    mindful-harness show              # opens an HTML render in the browser
    mindful-harness show --json       # prints state as JSON to stdout
    mindful-harness vitals
    mindful-harness alarms
    mindful-harness drift
    mindful-harness reset             # wipe state (with confirmation)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import webbrowser
from pathlib import Path

from mindful_harness import Conditional, FirehoseItem
from mindful_harness.drift import detect_drift
from mindful_harness.persistence import load_or_new, mind_session
from mindful_harness.viz import render_mind_html, render_mind_json

DEFAULT_STATE_PATH = Path(
    os.environ.get(
        "MINDFUL_HARNESS_STATE", str(Path.home() / ".mindful-harness" / "mind.json")
    )
)


def _state_path(args: argparse.Namespace) -> Path:
    return Path(args.state).expanduser() if args.state else DEFAULT_STATE_PATH


def _write_html_safely(path: Path, html: str) -> None:
    """Write an HTML export with 0600 permissions via atomic temp + replace.

    Refuses symlinked output paths so an attacker-planted symlink at
    `mind.html` cannot redirect the write onto an arbitrary user-owned
    file. Same defense as persistence.save() applies here because the
    rendered HTML can contain beliefs/questions/opportunities just as
    sensitive as the state file.
    """
    p_user = path.expanduser()
    if p_user.is_symlink():
        raise PermissionError(
            f"refusing to write HTML export to symlinked path: {p_user}. "
            "Remove the symlink or choose a non-symlinked path."
        )
    p = p_user.resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html)
        os.chmod(tmp, 0o600)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def cmd_status(args: argparse.Namespace) -> int:
    mind = load_or_new(_state_path(args))
    signs = mind.vital_signs()
    alarms = mind.certainty_alarms()
    print(f"State: {_state_path(args)}")
    print()
    print("Vital signs:")
    for name, value in signs.items():
        print(f"  {name:30s} {value:>8.2f}")
    if alarms:
        print()
        print(f"Certainty alarms ({len(alarms)}):")
        for name, c in alarms:
            print(f"  {name} (conf {c.confidence:.2f})")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    if args.from_stdin:
        content = sys.stdin.read()
    elif args.content:
        content = args.content
    else:
        print("ingest requires --content or --from-stdin", file=sys.stderr)
        return 2

    item = FirehoseItem(source=args.source, content=content)
    debug = os.environ.get("MINDFUL_HARNESS_DEBUG")
    state = _state_path(args)

    if not args.with_llm:
        # Fast path: no external call, single session.
        with mind_session(state) as mind:
            mind.ingest(item)
        print(f"Ingested item from {args.source} (no distillation).")
        return 0

    # LLM path: do NOT hold the state lock during the (potentially
    # 120s+) external call. Snapshot under lock for context, release
    # for the call, reacquire to apply.
    snapshot = load_or_new(state)

    from mindful_harness.llm import (
        apply_distillation,
        distill_item,
    )

    try:
        distillation = distill_item(item, snapshot, model=args.model)
        print(
            f"Distilled via {args.model} "
            f"({distillation.get('_cost_usd', 0):.4f} USD, "
            f"{distillation.get('_duration_ms', 0)/1000:.1f}s)"
        )
    except Exception as e:
        # No state was mutated under lock; safe to fall back.
        msg = f"LLM ingest failed: {type(e).__name__}"
        if debug:
            msg += f": {e}"
        print(msg, file=sys.stderr)
        print("Item added to firehose without distillation.", file=sys.stderr)
        with mind_session(state) as mind:
            mind.ingest(item)
        return 0

    # Apply phase: re-acquire the lock and check for keys the
    # snapshot did not have. For beliefs and knowledge, the
    # distillation might overwrite entries another concurrent writer
    # added in the meantime. Skip overwriting any belief/knowledge
    # key whose current confidence is higher than ours; otherwise
    # commit. This is best-effort conflict resolution; a stronger
    # design would attach versions to each entry.
    with mind_session(state) as mind:
        # Filter beliefs/knowledge against the live Mind to avoid
        # clobbering higher-confidence updates that arrived between
        # snapshot and apply.
        _drop_lower_confidence_overwrites(distillation, mind)
        apply_distillation(distillation, mind, item)
    return 0


def _drop_lower_confidence_overwrites(distillation: dict, live_mind) -> None:
    """In-place mutate distillation to skip belief/knowledge overwrites
    that would lower confidence relative to what the live Mind already
    has. Mitigates lost-update races on concurrent LLM ingests."""
    for kind in ("belief_updates", "knowledge_updates"):
        updates = distillation.get(kind, [])
        target = live_mind.beliefs if kind == "belief_updates" else live_mind.knowledge
        filtered = []
        for u in updates:
            key = (u.get("key") or "").strip()
            if key and key in target:
                live_conf = target[key].confidence
                new_conf = float(u.get("confidence", 0.5))
                if new_conf < live_conf:
                    continue  # don't clobber a higher-confidence concurrent write
            filtered.append(u)
        distillation[kind] = filtered


def cmd_believe(args: argparse.Namespace) -> int:
    if len(args.alts) < 2:
        print(
            "believe requires at least two alternatives (--alt ... --alt ...)",
            file=sys.stderr,
        )
        return 2

    c = Conditional(
        value=args.value,
        confidence=args.confidence,
        alternatives=args.alts,
        framing=args.framing or "",
        reversion_trigger=args.revise_if,
    )
    with mind_session(_state_path(args)) as mind:
        mind.believe(args.key, c)
    print(f"Set belief {args.key!r}: could be {args.value!r} (conf {args.confidence:.2f})")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    with mind_session(_state_path(args)) as mind:
        q = mind.ask(args.text, routed_to=args.routed_to)
    print(f"Opened question: {q.text}")
    return 0


def cmd_notice(args: argparse.Namespace) -> int:
    with mind_session(_state_path(args)) as mind:
        i = mind.notice(args.text)
    print(f"Noticed: {i.text} (intensity {i.intensity:.2f})")
    return 0


def cmd_wonder(args: argparse.Namespace) -> int:
    with mind_session(_state_path(args)) as mind:
        c = mind.wonder(args.text)
    print(f"Wondering: {c.text}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    mind = load_or_new(_state_path(args))
    if args.json:
        print(render_mind_json(mind))
        return 0
    html = render_mind_html(mind, title=args.title or "Mind")
    output = Path(args.output) if args.output else Path.cwd() / "mind.html"
    _write_html_safely(output, html)
    print(f"Rendered to {output.resolve()}")
    if args.open:
        webbrowser.open(f"file://{output.resolve()}")
    return 0


def cmd_vitals(args: argparse.Namespace) -> int:
    mind = load_or_new(_state_path(args))
    for name, value in mind.vital_signs().items():
        print(f"{name:30s} {value:>8.2f}")
    return 0


def cmd_alarms(args: argparse.Namespace) -> int:
    mind = load_or_new(_state_path(args))
    alarms = mind.certainty_alarms()
    if not alarms:
        print("No certainty alarms.")
        return 0
    for name, c in alarms:
        print(f"  {name}: conf {c.confidence:.2f}, value could be {c.value!r}")
        if c.alternatives:
            print(f"    alternatives: {', '.join(str(a) for a in c.alternatives)}")
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    mind = load_or_new(_state_path(args))
    report = detect_drift(mind)
    print(f"Stale beliefs: {len(report.stale_beliefs)}")
    for key, age in report.stale_beliefs:
        days = age / 86400.0
        print(f"  {key}: {days:.1f} days old")
    print(f"Stale knowledge: {len(report.stale_knowledge)}")
    for key, age in report.stale_knowledge:
        days = age / 86400.0
        print(f"  {key}: {days:.1f} days old")
    print(f"Stale decisions: {len(report.stale_decisions)}")
    for chosen, age in report.stale_decisions:
        days = age / 86400.0
        print(f"  {chosen}: {days:.1f} days old")
    print(f"Aging questions: {len(report.aging_questions)}")
    for text, age in report.aging_questions:
        days = age / 86400.0
        print(f"  {text}: {days:.1f} days old")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    path = _state_path(args)
    if not path.exists():
        print(f"No state at {path}; nothing to reset.")
        return 0
    if not args.yes:
        reply = input(f"Delete Mind state at {path}? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return 1
    path.unlink()
    print(f"Removed {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mindful-harness",
        description="A mindful substrate for AI-native work, inspired by Ellen Langer.",
    )
    parser.add_argument(
        "--state",
        help=f"Path to Mind state file (default: {DEFAULT_STATE_PATH}).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Quick view of vital signs and alarms.")
    p_status.set_defaults(func=cmd_status)

    p_ingest = sub.add_parser("ingest", help="Ingest a firehose item.")
    p_ingest.add_argument("--source", required=True, help="Source label (email, rss, etc.).")
    p_ingest.add_argument("--content", help="Item content as a single string.")
    p_ingest.add_argument("--from-stdin", action="store_true", help="Read content from stdin.")
    p_ingest.add_argument("--with-llm", action="store_true", help="Distill via Claude CLI.")
    p_ingest.add_argument("--model", default="haiku", help="Model for distillation.")
    p_ingest.set_defaults(func=cmd_ingest)

    p_believe = sub.add_parser("believe", help="Set or update a belief.")
    p_believe.add_argument("key", help="Short stable identifier for the belief.")
    p_believe.add_argument("value", help="The proposed claim (in conditional language).")
    p_believe.add_argument("--alt", action="append", dest="alts", default=[],
                           help="An alternative framing (use at least twice).")
    p_believe.add_argument("--confidence", type=float, default=0.5,
                           help="Confidence in [0, 1]. Default 0.5.")
    p_believe.add_argument("--framing", help="The lens under which this belief holds.")
    p_believe.add_argument("--revise-if", dest="revise_if",
                           help="Condition that would trigger revision.")
    p_believe.set_defaults(func=cmd_believe)

    p_ask = sub.add_parser("ask", help="Open a question.")
    p_ask.add_argument("text", help="The question.")
    p_ask.add_argument("--routed-to", choices=["firehose", "hands", "human"],
                       help="Where this question should be routed.")
    p_ask.set_defaults(func=cmd_ask)

    p_notice = sub.add_parser("notice", help="Note an interest (anomaly, deviation).")
    p_notice.add_argument("text", help="The observation.")
    p_notice.set_defaults(func=cmd_notice)

    p_wonder = sub.add_parser("wonder", help="Note a curiosity (directed thread to follow).")
    p_wonder.add_argument("text", help="The thread.")
    p_wonder.set_defaults(func=cmd_wonder)

    p_show = sub.add_parser("show", help="Render Mind state to HTML or JSON.")
    p_show.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    p_show.add_argument("--output", help="Where to write the HTML (default ./mind.html).")
    p_show.add_argument("--title", help="Title for the HTML page.")
    p_show.add_argument("--open", action="store_true", help="Open the rendered HTML in a browser.")
    p_show.set_defaults(func=cmd_show)

    p_vitals = sub.add_parser("vitals", help="Print vital signs.")
    p_vitals.set_defaults(func=cmd_vitals)

    p_alarms = sub.add_parser("alarms", help="Print certainty alarms.")
    p_alarms.set_defaults(func=cmd_alarms)

    p_drift = sub.add_parser("drift", help="Run drift detection on aging state.")
    p_drift.set_defaults(func=cmd_drift)

    p_reset = sub.add_parser("reset", help="Delete the Mind state file.")
    p_reset.add_argument("--yes", action="store_true", help="Skip confirmation.")
    p_reset.set_defaults(func=cmd_reset)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
