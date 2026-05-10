"""Persisting and restoring Mind state.

The Mind is meant to be long-running, so its state must survive across
process restarts. v0.0.x uses a single JSON file as the backing store:
the Mind serializes on `save` and rehydrates on `load`. Eventually this
will become a richer store (event log, durable queue), but the simple
file is enough to be useful today.

Serialization is via the same shape as `viz.render_mind_json`, which
makes the persisted file directly readable for inspection or for
feeding into other tools.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import fcntl  # POSIX only
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from mindful_harness.mind import (
    Curiosity,
    FirehoseItem,
    Idea,
    Interest,
    Mind,
    Opportunity,
    Question,
)
from mindful_harness.primitives import Conditional, Decision, Distinction


def _serialize_conditional(c: Conditional[Any]) -> dict[str, Any]:
    return {
        "value": c.value,
        "confidence": c.confidence,
        "alternatives": list(c.alternatives),
        "framing": c.framing,
        "reversion_trigger": c.reversion_trigger,
    }


def _deserialize_conditional(d: dict[str, Any]) -> Conditional[Any]:
    return Conditional(
        value=d["value"],
        confidence=float(d["confidence"]),
        alternatives=list(d.get("alternatives", [])),
        framing=d.get("framing", ""),
        reversion_trigger=d.get("reversion_trigger"),
    )


def to_dict(mind: Mind) -> dict[str, Any]:
    """Serialize a Mind to a plain dict suitable for JSON encoding."""
    return {
        "version": 1,
        "saved_at": time.time(),
        "beliefs": {k: _serialize_conditional(c) for k, c in mind.beliefs.items()},
        "knowledge": {k: _serialize_conditional(c) for k, c in mind.knowledge.items()},
        "questions": [
            {
                "text": q.text,
                "born_at": q.born_at,
                "routed_to": q.routed_to,
                "resolved": q.resolved,
                "resolution": q.resolution,
            }
            for q in mind.questions
        ],
        "interests": [
            {
                "text": i.text,
                "born_at": i.born_at,
                "last_reinforced": i.last_reinforced,
                "intensity": i.intensity,
            }
            for i in mind.interests
        ],
        "curiosities": [
            {"text": c.text, "born_at": c.born_at, "pursued": c.pursued}
            for c in mind.curiosities
        ],
        "opportunities": [
            {
                "text": o.text,
                "enables": o.enables,
                "born_at": o.born_at,
                "acted_on": o.acted_on,
            }
            for o in mind.opportunities
        ],
        "ideas": [
            {"text": i.text, "connects": list(i.connects), "born_at": i.born_at}
            for i in mind.ideas
        ],
        "decisions": [
            {
                "chosen": str(d.chosen),
                "rejected": [str(r) for r in d.rejected],
                "framing": d.framing,
                "reversion_triggers": list(d.reversion_triggers),
                "confidence": d.confidence,
                "timestamp": d.timestamp,
            }
            for d in mind.decisions
        ],
        "ingestion_log": [
            {
                "source": item.source,
                "content": str(item.content),
                "timestamp": item.timestamp,
                "meta": item.meta,
            }
            for item in mind.ingestion_log
        ],
        "distinction_log": [
            {
                "compared_to": str(d.compared_to),
                "noticed": d.noticed,
                "forces_new_category": d.forces_new_category,
            }
            for d in mind.distinction_log
        ],
        "action_history": {k: list(v) for k, v in mind.action_history.items()},
    }


def from_dict(data: dict[str, Any]) -> Mind:
    """Rehydrate a Mind from a previously-serialized dict."""
    if data.get("version") != 1:
        raise ValueError(
            f"Unsupported Mind persistence version: {data.get('version')!r}"
        )

    mind = Mind()

    for key, c in data.get("beliefs", {}).items():
        mind.beliefs[key] = _deserialize_conditional(c)

    for key, c in data.get("knowledge", {}).items():
        mind.knowledge[key] = _deserialize_conditional(c)

    for q in data.get("questions", []):
        question = Question(
            text=q["text"],
            born_at=q.get("born_at", time.time()),
            routed_to=q.get("routed_to"),
            resolved=q.get("resolved", False),
            resolution=q.get("resolution"),
        )
        mind.questions.append(question)

    for i in data.get("interests", []):
        interest = Interest(
            text=i["text"],
            born_at=i.get("born_at", time.time()),
            last_reinforced=i.get("last_reinforced", time.time()),
            intensity=i.get("intensity", 1.0),
        )
        mind.interests.append(interest)

    for c in data.get("curiosities", []):
        mind.curiosities.append(
            Curiosity(
                text=c["text"],
                born_at=c.get("born_at", time.time()),
                pursued=c.get("pursued", False),
            )
        )

    for o in data.get("opportunities", []):
        mind.opportunities.append(
            Opportunity(
                text=o["text"],
                enables=o.get("enables", ""),
                born_at=o.get("born_at", time.time()),
                acted_on=o.get("acted_on", False),
            )
        )

    for i in data.get("ideas", []):
        mind.ideas.append(
            Idea(
                text=i["text"],
                connects=list(i.get("connects", [])),
                born_at=i.get("born_at", time.time()),
            )
        )

    for d in data.get("decisions", []):
        decision = Decision(
            chosen=d["chosen"],
            rejected=list(d["rejected"]),
            framing=d.get("framing", ""),
            reversion_triggers=list(d["reversion_triggers"]),
            confidence=float(d["confidence"]),
            timestamp=d.get("timestamp", time.time()),
        )
        mind.decisions.append(decision)

    for item in data.get("ingestion_log", []):
        mind.ingestion_log.append(
            FirehoseItem(
                source=item["source"],
                content=item.get("content", ""),
                timestamp=item.get("timestamp", time.time()),
                meta=item.get("meta", {}),
            )
        )

    for dist in data.get("distinction_log", []):
        try:
            mind.distinction_log.append(
                Distinction(
                    item=None,
                    compared_to=dist["compared_to"],
                    noticed=dist["noticed"],
                    forces_new_category=dist.get("forces_new_category", False),
                )
            )
        except ValueError:
            # Skip malformed distinctions on rehydrate.
            continue

    for sig, timestamps in data.get("action_history", {}).items():
        mind.action_history[sig] = list(timestamps)

    return mind


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    """Best-effort exclusive lock across processes on POSIX systems.

    Falls back to a no-op on platforms without fcntl. Used so concurrent
    CLI invocations cannot interleave load/mutate/save sequences and
    lose updates.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(lock_path, "w")  # noqa: SIM115 - lifetime managed manually
    try:
        if fcntl is not None:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass  # filesystem doesn't support locks; continue best-effort
        yield
    finally:
        try:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        finally:
            lock_f.close()
            with contextlib.suppress(OSError):
                lock_path.unlink()


def _restrict_dir_permissions(directory: Path) -> None:
    """Force the state directory to 0700 if we own it. Best effort."""
    with contextlib.suppress(OSError):
        if directory.stat().st_uid == os.getuid():
            os.chmod(directory, 0o700)


def save(mind: Mind, path: str | Path) -> Path:
    """Save a Mind to a JSON file atomically with restrictive permissions.

    The file is written to a sibling temp file with mode 0600, then
    atomically renamed onto the target path. A `.lock` file in the
    parent directory serializes concurrent writers so two CLI runs
    cannot lose each other's updates. Returns the resolved path.
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    _restrict_dir_permissions(p.parent)

    payload = json.dumps(to_dict(mind), indent=2, default=str).encode("utf-8")

    lock_path = p.with_name(p.name + ".lock")
    with _file_lock(lock_path):
        fd, tmp_path_str = tempfile.mkstemp(
            dir=p.parent, prefix=f".{p.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "wb") as tmp_f:
                tmp_f.write(payload)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, p)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            raise

    return p


def load(path: str | Path) -> Mind:
    """Load a Mind from a JSON file. Raises FileNotFoundError if missing."""
    p = Path(path).expanduser().resolve()
    lock_path = p.with_name(p.name + ".lock")
    if p.parent.exists():
        with _file_lock(lock_path):
            return from_dict(json.loads(p.read_text()))
    return from_dict(json.loads(p.read_text()))


def load_or_new(path: str | Path) -> Mind:
    """Load a Mind from disk if the file exists, else return a fresh Mind."""
    p = Path(path).expanduser()
    if p.exists():
        return load(p)
    return Mind()
