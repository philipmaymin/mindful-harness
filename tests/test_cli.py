"""Tests for the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindful_harness.cli import main


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "mind.json"


def run(state_path: Path, *args: str, stdin: str | None = None, monkeypatch=None) -> int:
    cli_args = ["--state", str(state_path), *args]
    if stdin is not None and monkeypatch is not None:
        import io
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    return main(cli_args)


class TestBelieveAndStatus:
    def test_believe_persists_and_status_reflects(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = run(
            state_path,
            "believe",
            "revenue",
            "up 12%",
            "--alt",
            "flat",
            "--alt",
            "down",
            "--confidence",
            "0.7",
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "revenue" in out
        assert state_path.exists()

        rc = run(state_path, "status")
        assert rc == 0
        out = capsys.readouterr().out
        assert "alternative_cardinality" in out

    def test_believe_rejects_fewer_than_two_alts(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = run(state_path, "believe", "x", "v", "--alt", "only-one")
        assert rc == 2
        err = capsys.readouterr().err
        assert "two alternatives" in err


class TestAskAndNotice:
    def test_ask_opens_question(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = run(state_path, "ask", "what drove Q1?")
        assert rc == 0
        capsys.readouterr()

        rc = run(state_path, "show", "--json")
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert any("what drove Q1?" in q["text"] for q in data["questions"])

    def test_notice_creates_interest(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = run(state_path, "notice", "support load dropped")
        assert rc == 0
        capsys.readouterr()

        rc = run(state_path, "show", "--json")
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert any("support load" in i["text"] for i in data["interests"])


class TestIngest:
    def test_ingest_with_content(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = run(
            state_path,
            "ingest",
            "--source",
            "email",
            "--content",
            "Vendor announced API",
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Ingested" in out

        rc = run(state_path, "show", "--json")
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # Ingestion log not directly serialized by render_mind_json; check vital signs instead.
        assert data["vital_signs"]["items_ingested"] == 1.0

    def test_ingest_requires_content_or_stdin(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = run(state_path, "ingest", "--source", "email")
        assert rc == 2
        err = capsys.readouterr().err
        assert "content" in err


class TestShow:
    def test_show_json_outputs_structure(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        run(state_path, "believe", "x", "A", "--alt", "B", "--alt", "C")
        capsys.readouterr()
        rc = run(state_path, "show", "--json")
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "x" in data["beliefs"]

    def test_show_html_writes_file(
        self, state_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        output = tmp_path / "out.html"
        rc = run(state_path, "show", "--output", str(output))
        assert rc == 0
        assert output.exists()
        content = output.read_text()
        assert "<!doctype html>" in content


class TestVitalsAndAlarms:
    def test_vitals(self, state_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = run(state_path, "vitals")
        assert rc == 0
        out = capsys.readouterr().out
        assert "items_ingested" in out

    def test_alarms_empty(self, state_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = run(state_path, "alarms")
        assert rc == 0
        out = capsys.readouterr().out
        assert "No certainty alarms" in out

    def test_alarms_with_high_confidence_belief(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        run(
            state_path,
            "believe",
            "x",
            "A",
            "--alt",
            "B",
            "--alt",
            "C",
            "--confidence",
            "0.95",
        )
        capsys.readouterr()
        rc = run(state_path, "alarms")
        assert rc == 0
        out = capsys.readouterr().out
        assert "x" in out
        assert "0.95" in out


class TestReset:
    def test_reset_with_yes(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        run(state_path, "believe", "x", "v", "--alt", "a", "--alt", "b")
        capsys.readouterr()
        assert state_path.exists()

        rc = run(state_path, "reset", "--yes")
        assert rc == 0
        assert not state_path.exists()

    def test_reset_when_no_state(
        self, state_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = run(state_path, "reset", "--yes")
        assert rc == 0
        out = capsys.readouterr().out
        assert "nothing to reset" in out
