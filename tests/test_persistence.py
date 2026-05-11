"""Tests for Mind persistence (save/load round-trip)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindful_harness import (
    Conditional,
    Decision,
    Distinction,
    FirehoseItem,
    Mind,
)
from mindful_harness.persistence import (
    from_dict,
    load,
    load_or_new,
    mind_session,
    save,
    to_dict,
)


@pytest.fixture
def populated_mind() -> Mind:
    m = Mind()
    m.believe(
        "revenue",
        Conditional(
            value="up 12%",
            confidence=0.7,
            alternatives=["flat", "down"],
            framing="initial Q1 review",
            reversion_trigger="if data refresh failed",
        ),
    )
    m.know(
        "api_announced",
        Conditional(value=True, confidence=0.95, alternatives=[False, "rumor"]),
    )
    m.ask("why did Q1 lift?", routed_to="hands")
    m.notice("support load dropped while NPS rose")
    m.wonder("Tuesday vs Monday cohort?")
    m.see_opportunity(text="parallel test API", enables="empirical risk assessment")
    m.connect(
        text="resembles the chambermaid effect",
        connects=["revenue", "chambermaid-study"],
    )
    m.commit(
        Decision(
            chosen="investigate",
            rejected=["assume seasonal", "assume noise"],
            framing="Q1 review",
            reversion_triggers=["if Q2 reverses"],
            confidence=0.6,
        )
    )
    m.ingest(FirehoseItem(source="email", content="hello"))
    m.record_distinction(
        Distinction(item="X", compared_to="prior X", noticed="X moved up")
    )
    m.record_action("send-weekly-report")
    m.record_action("send-weekly-report")
    m.record_action("send-weekly-report")
    return m


class TestSerialization:
    def test_to_dict_has_version(self, populated_mind: Mind) -> None:
        d = to_dict(populated_mind)
        assert d["version"] == 1

    def test_to_dict_is_json_serializable(self, populated_mind: Mind) -> None:
        d = to_dict(populated_mind)
        text = json.dumps(d, default=str)
        # Round-trip valid JSON
        json.loads(text)

    def test_roundtrip_preserves_beliefs(self, populated_mind: Mind) -> None:
        d = to_dict(populated_mind)
        restored = from_dict(d)
        assert "revenue" in restored.beliefs
        assert restored.beliefs["revenue"].value == "up 12%"
        assert restored.beliefs["revenue"].confidence == 0.7
        assert restored.beliefs["revenue"].alternatives == ["flat", "down"]
        assert restored.beliefs["revenue"].reversion_trigger == "if data refresh failed"

    def test_roundtrip_preserves_knowledge(self, populated_mind: Mind) -> None:
        d = to_dict(populated_mind)
        restored = from_dict(d)
        assert "api_announced" in restored.knowledge
        assert restored.knowledge["api_announced"].confidence == 0.95

    def test_roundtrip_preserves_questions(self, populated_mind: Mind) -> None:
        d = to_dict(populated_mind)
        restored = from_dict(d)
        assert len(restored.questions) == 1
        assert restored.questions[0].text == "why did Q1 lift?"
        assert restored.questions[0].routed_to == "hands"

    def test_roundtrip_preserves_decisions(self, populated_mind: Mind) -> None:
        d = to_dict(populated_mind)
        restored = from_dict(d)
        assert len(restored.decisions) == 1
        assert restored.decisions[0].chosen == "investigate"
        assert restored.decisions[0].rejected == ["assume seasonal", "assume noise"]
        assert "if Q2 reverses" in restored.decisions[0].reversion_triggers

    def test_roundtrip_preserves_opportunities(self, populated_mind: Mind) -> None:
        d = to_dict(populated_mind)
        restored = from_dict(d)
        assert len(restored.opportunities) == 1
        assert "parallel test" in restored.opportunities[0].text

    def test_roundtrip_preserves_ideas(self, populated_mind: Mind) -> None:
        d = to_dict(populated_mind)
        restored = from_dict(d)
        assert len(restored.ideas) == 1
        assert "chambermaid-study" in restored.ideas[0].connects

    def test_roundtrip_preserves_habituation(self, populated_mind: Mind) -> None:
        d = to_dict(populated_mind)
        restored = from_dict(d)
        assert restored.is_habituated("send-weekly-report")

    def test_unsupported_version_raises(self) -> None:
        with pytest.raises(ValueError, match="version"):
            from_dict({"version": 99})


class TestFileIO:
    def test_save_and_load(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        path = tmp_path / "mind.json"
        save(populated_mind, path)
        assert path.exists()
        restored = load(path)
        assert "revenue" in restored.beliefs

    def test_save_creates_parent_dirs(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        path = tmp_path / "nested" / "deeper" / "mind.json"
        save(populated_mind, path)
        assert path.exists()

    def test_load_or_new_returns_empty_when_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "does_not_exist.json"
        mind = load_or_new(path)
        assert isinstance(mind, Mind)
        assert mind.beliefs == {}

    def test_load_or_new_loads_when_present(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        path = tmp_path / "mind.json"
        save(populated_mind, path)
        mind = load_or_new(path)
        assert "revenue" in mind.beliefs


class TestFileSecurityAndAtomicity:
    def test_state_file_has_0600_permissions(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        import os
        import stat
        path = tmp_path / "secrets" / "mind.json"
        save(populated_mind, path)
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"

    def test_state_directory_has_0700_permissions(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        import os
        import stat
        path = tmp_path / "secrets" / "mind.json"
        save(populated_mind, path)
        mode = stat.S_IMODE(os.stat(path.parent).st_mode)
        assert mode == 0o700, f"expected 0700, got {oct(mode)}"

    def test_save_leaves_no_tmp_files_on_success(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        path = tmp_path / "mind.json"
        save(populated_mind, path)
        # Lock files are intentionally left in place so concurrent
        # writers cannot race on lock-file creation (symlink-safe fix).
        # We only assert that no .tmp files leak.
        leftovers = [
            p for p in tmp_path.iterdir() if p.name.endswith(".tmp")
        ]
        assert leftovers == []

    def test_save_is_atomic_on_failure(
        self, populated_mind: Mind, tmp_path: Path, monkeypatch
    ) -> None:
        """If the write fails mid-flight, the previous file should be intact."""
        path = tmp_path / "mind.json"
        save(populated_mind, path)
        original_bytes = path.read_bytes()

        # Force the temp-file write to fail.
        import mindful_harness.persistence as persistence_mod

        def boom(src, dst):
            raise RuntimeError("simulated write failure")

        monkeypatch.setattr(persistence_mod.os, "replace", boom)

        with pytest.raises(RuntimeError):
            # Mutate so the save would otherwise change the file.
            populated_mind.notice("a new anomaly to persist")
            save(populated_mind, path)

        # File on disk is the original, not partial.
        assert path.read_bytes() == original_bytes

    def test_preexisting_parent_directory_not_chmodded(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        """save() must not silently lock down a parent directory the user already had."""
        import os
        import stat
        # tmp_path is created by pytest; treat it as the user's shared dir.
        original_mode = stat.S_IMODE(os.stat(tmp_path).st_mode)
        path = tmp_path / "mind.json"
        save(populated_mind, path)
        new_mode = stat.S_IMODE(os.stat(tmp_path).st_mode)
        assert new_mode == original_mode, (
            f"save() unexpectedly chmodded a pre-existing parent: {oct(original_mode)} -> {oct(new_mode)}"
        )

    def test_newly_created_parent_directory_is_chmodded_0700(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        """save() creates its own dedicated state directory at 0700."""
        import os
        import stat
        path = tmp_path / "new-state-dir" / "mind.json"
        assert not path.parent.exists()
        save(populated_mind, path)
        mode = stat.S_IMODE(os.stat(path.parent).st_mode)
        assert mode == 0o700


class TestLockSafety:
    def test_lock_failure_raises_does_not_silently_proceed(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        """If the lock file is a symlink, save() must raise — never silently bypass lock.

        Regression test for the v0.0.9 bug where _file_lock fell through to an
        unlocked yield on os.open failure.
        """
        from mindful_harness.persistence import LockAcquisitionError

        path = tmp_path / "mind.json"
        # Plant a symlink at the expected lock path before any save happens.
        lock_path = path.with_name(path.name + ".lock")
        decoy = tmp_path / "decoy.txt"
        decoy.write_text("untouched")
        lock_path.symlink_to(decoy)

        with pytest.raises(LockAcquisitionError):
            save(populated_mind, path)
        # Decoy was NOT clobbered by the lock attempt.
        assert decoy.read_text() == "untouched"

    def test_refuses_symlinked_state_path(
        self, populated_mind: Mind, tmp_path: Path
    ) -> None:
        """save() must not follow a symlinked state path onto an arbitrary target."""
        real_path = tmp_path / "real.json"
        real_path.write_text("untouched")
        symlink_path = tmp_path / "mind.json"
        symlink_path.symlink_to(real_path)

        with pytest.raises(PermissionError, match="symlinked state path"):
            save(populated_mind, symlink_path)
        # Real target was not clobbered.
        assert real_path.read_text() == "untouched"


class TestMindSession:
    def test_session_persists_changes_on_normal_exit(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "mind.json"
        with mind_session(path) as mind:
            mind.ask("first question")
        # Reload from disk and confirm.
        mind = load_or_new(path)
        assert any("first question" == q.text for q in mind.questions)

    def test_session_skips_save_on_exception(self, tmp_path: Path) -> None:
        path = tmp_path / "mind.json"
        with mind_session(path) as mind:
            mind.ask("committed")
        # First session committed.
        with pytest.raises(RuntimeError):
            with mind_session(path) as mind:
                mind.ask("uncommitted")
                raise RuntimeError("bail out before save")
        # The uncommitted question is NOT persisted.
        mind = load_or_new(path)
        texts = {q.text for q in mind.questions}
        assert "committed" in texts
        assert "uncommitted" not in texts
