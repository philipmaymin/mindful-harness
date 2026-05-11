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
        "last_reviewed": c.last_reviewed,
    }


def _deserialize_conditional(d: dict[str, Any]) -> Conditional[Any]:
    c = Conditional(
        value=d["value"],
        confidence=float(d["confidence"]),
        alternatives=list(d.get("alternatives", [])),
        framing=d.get("framing", ""),
        reversion_trigger=d.get("reversion_trigger"),
    )
    if "last_reviewed" in d:
        c.last_reviewed = float(d["last_reviewed"])
    return c


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


class LockAcquisitionError(OSError):
    """Raised when the persistence lock cannot be safely acquired.

    Includes the symlink-attack case: if `mind.json.lock` is a symlink,
    O_NOFOLLOW makes `os.open` refuse to follow it, and we raise rather
    than fall through to an unlocked write.
    """


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    """Exclusive lock across processes on POSIX systems.

    The lock file is opened with O_NOFOLLOW on POSIX so a symlink at the
    lock path cannot be followed to truncate an attacker-chosen
    arbitrary file. If the open fails (symlink, permission, full disk,
    etc.) the function raises `LockAcquisitionError` rather than
    silently proceeding without a lock — silent-fallback was the bug
    audit pass #3 caught in v0.0.9.

    The lock file is left in place after release so concurrent writers
    cannot race on its creation.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(lock_path), flags, 0o600)
    except OSError as e:
        raise LockAcquisitionError(
            f"refusing to acquire lock at {lock_path}: {e.__class__.__name__}. "
            "If the lock path is a symlink, remove it. "
            "If the directory is unwritable, choose a different --state path."
        ) from e
    lock_f = os.fdopen(fd, "r+b")
    locked = False
    try:
        if fcntl is not None:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                locked = True
            except OSError:
                # Filesystem (e.g. NFS without lockd) does not support
                # advisory locks. Continue without locking; the
                # exclusive O_CREAT still provides some protection.
                pass
        yield
    finally:
        try:
            if locked and fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        finally:
            lock_f.close()


def _refuse_symlinked_target(p_user: Path) -> None:
    """Refuse to write through a symlinked state file.

    Must be called on the user-supplied (un-resolved) path, since
    `.resolve()` would already have followed the symlink. Detects the
    case where `--state ~/innocent.json` is a symlink to an
    attacker-chosen target.
    """
    if p_user.is_symlink():
        raise PermissionError(
            f"refusing to write to symlinked state path: {p_user}. "
            "Remove the symlink or choose a non-symlinked path."
        )


def _restrict_dir_permissions(directory: Path) -> None:
    """Force the state directory to 0700 if we own it. Best effort."""
    with contextlib.suppress(OSError):
        if directory.stat().st_uid == os.getuid():
            os.chmod(directory, 0o700)


def _save_locked(mind: Mind, p: Path) -> None:
    """Write Mind to p atomically. Caller must already hold the path lock.

    Symlink check is performed by the public `save()` / `mind_session()`
    entry points on the user-supplied (pre-resolve) path.
    """
    payload = json.dumps(to_dict(mind), indent=2, default=str).encode("utf-8")
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


def save(mind: Mind, path: str | Path) -> Path:
    """Save a Mind to a JSON file atomically with restrictive permissions.

    The file is written to a sibling temp file with mode 0600, then
    atomically renamed onto the target path. A `.lock` file in the
    parent directory serializes concurrent writers so two CLI runs
    cannot lose each other's updates.

    The parent directory is chmod'd to 0700 ONLY if this call had to
    create it. Pre-existing user directories are left as-is so a
    `--state ~/shared/mind.json` does not silently lock other
    collaborators out of `~/shared`.

    A symlinked state path is refused before any I/O, so an attacker
    cannot redirect the write onto an arbitrary user-owned file.

    Returns the resolved path.
    """
    p_user = Path(path).expanduser()
    _refuse_symlinked_target(p_user)
    p = p_user.resolve()
    parent_existed = p.parent.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed:
        _restrict_dir_permissions(p.parent)

    lock_path = p.with_name(p.name + ".lock")
    with _file_lock(lock_path):
        _save_locked(mind, p)

    return p


def _load_locked(p: Path) -> Mind:
    """Read Mind from p. Caller must already hold the path lock if it exists.

    Symlink check is performed by the public `load()` / `mind_session()`
    entry points on the user-supplied (pre-resolve) path.
    """
    return from_dict(json.loads(p.read_text()))


def load(path: str | Path) -> Mind:
    """Load a Mind from a JSON file. Raises FileNotFoundError if missing.

    A symlinked state path is refused before any I/O.
    """
    p_user = Path(path).expanduser()
    _refuse_symlinked_target(p_user)
    p = p_user.resolve()
    lock_path = p.with_name(p.name + ".lock")
    if p.parent.exists():
        with _file_lock(lock_path):
            return _load_locked(p)
    return _load_locked(p)


def load_or_new(path: str | Path) -> Mind:
    """Load a Mind from disk if the file exists, else return a fresh Mind."""
    p = Path(path).expanduser()
    if p.exists():
        return load(p)
    return Mind()


@contextlib.contextmanager
def mind_session(path: str | Path):
    """Yield a Mind, persist it back under a single lock on exit.

    Wraps load-mutate-save inside one acquisition of the path lock so
    concurrent CLI invocations cannot interleave and lose each other's
    updates. The Mind is loaded from disk on entry (or fresh if no file
    exists) and saved on normal exit. On exception, the original file
    is left intact (the lock is released without saving).

    Usage::

        with mind_session("~/.mindful-harness/mind.json") as mind:
            mind.believe(...)
            mind.ask(...)
        # state is now persisted atomically with the changes applied
    """
    p_user = Path(path).expanduser()
    _refuse_symlinked_target(p_user)
    p = p_user.resolve()
    parent_existed = p.parent.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed:
        _restrict_dir_permissions(p.parent)

    lock_path = p.with_name(p.name + ".lock")
    with _file_lock(lock_path):
        mind = _load_locked(p) if p.exists() else Mind()
        yield mind
        _save_locked(mind, p)
