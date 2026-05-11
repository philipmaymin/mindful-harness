"""Tests for the LLM integration.

The integration shells out to `claude -p`, which is expensive to run in
CI. These tests mock subprocess.run and verify the prompt construction,
schema handling, envelope parsing, and Mind mutation logic. A live
end-to-end test lives in `examples/founders_inbox.py`.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mindful_harness import Conditional, FirehoseItem, Mind
from mindful_harness.llm import (
    DISTILLATION_SCHEMA,
    HAND_SCHEMA,
    SYSTEM_PROMPT,
    ClaudeCLIError,
    _run_claude,
    _summarize_mind,
    _user_prompt,
    apply_distillation,
    distill_item,
    execute_hand_llm,
)


@pytest.fixture
def empty_mind() -> Mind:
    return Mind()


@pytest.fixture
def populated_mind() -> Mind:
    m = Mind()
    m.believe(
        "x",
        Conditional(
            value="X is up", confidence=0.7, alternatives=["X is flat", "X is down"]
        ),
    )
    m.ask("why X?")
    m.notice("X moved without a known cause")
    return m


@pytest.fixture
def sample_item() -> FirehoseItem:
    return FirehoseItem(source="email", content="An interesting message")


@pytest.fixture
def well_formed_distillation() -> dict:
    return {
        "framework": "test framework",
        "not_looking_for": "test blindspots",
        "distinctions": [
            {"compared_to": "prior X", "noticed": "this is different", "forces_new_category": False}
        ],
        "belief_updates": [
            {
                "key": "test_belief",
                "value": "could be A",
                "confidence": 0.6,
                "alternatives": ["could be B", "could be C"],
            }
        ],
        "knowledge_updates": [],
        "questions": ["what about Y?"],
        "interests": ["something deviant"],
        "curiosities": [],
        "opportunities": [{"text": "do thing", "enables": "good outcome for counterpart"}],
        "ideas": [{"text": "X resembles Y", "connects": ["X-domain", "Y-domain"]}],
        "self_check": "could be wrong because Z",
    }


class TestSummarizeMind:
    def test_empty_mind(self, empty_mind: Mind) -> None:
        summary = _summarize_mind(empty_mind)
        assert summary == "(Mind is empty.)"

    def test_populated_mind_includes_beliefs(self, populated_mind: Mind) -> None:
        summary = _summarize_mind(populated_mind)
        assert "x" in summary
        assert "could be 'X is up'" in summary

    def test_populated_mind_includes_open_questions(
        self, populated_mind: Mind
    ) -> None:
        summary = _summarize_mind(populated_mind)
        assert "why X?" in summary

    def test_resolved_questions_excluded(self) -> None:
        m = Mind()
        q = m.ask("resolved?")
        q.resolved = True
        m.ask("still open?")
        summary = _summarize_mind(m)
        assert "still open?" in summary
        assert "resolved?" not in summary


class TestUserPrompt:
    def test_includes_source(self, sample_item: FirehoseItem) -> None:
        prompt = _user_prompt(sample_item, "(Mind is empty.)")
        assert "email" in prompt
        assert "An interesting message" in prompt

    def test_includes_mind_summary(self, sample_item: FirehoseItem) -> None:
        prompt = _user_prompt(sample_item, "Recent beliefs: ...")
        assert "Recent beliefs" in prompt

    def test_long_content_truncated(self) -> None:
        long_content = "a" * 5000
        item = FirehoseItem(source="rss", content=long_content)
        prompt = _user_prompt(item, "(empty)")
        assert "truncated" in prompt
        assert len(prompt) < len(long_content) + 1000


class TestSchema:
    def test_schema_has_required_fields(self) -> None:
        required = DISTILLATION_SCHEMA.get("required", [])
        assert "framework" in required
        assert "not_looking_for" in required

    def test_belief_updates_minimum_alternatives(self) -> None:
        belief_schema = DISTILLATION_SCHEMA["properties"]["belief_updates"]["items"]
        alt_schema = belief_schema["properties"]["alternatives"]
        assert alt_schema["minItems"] == 2  # plus the value = 3 framings total

    def test_confidence_bounded(self) -> None:
        belief_schema = DISTILLATION_SCHEMA["properties"]["belief_updates"]["items"]
        conf = belief_schema["properties"]["confidence"]
        assert conf["minimum"] == 0
        assert conf["maximum"] == 1


def _envelope(structured: dict) -> dict:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "",
        "duration_ms": 1234,
        "total_cost_usd": 0.01,
        "structured_output": structured,
    }


def _completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


class TestRunClaude:
    @patch("mindful_harness.llm._run_subprocess_bounded")
    def test_passes_required_args(
        self, mock_run: MagicMock, well_formed_distillation: dict
    ) -> None:
        mock_run.return_value = _completed_process(
            json.dumps(_envelope(well_formed_distillation))
        )
        _run_claude(user_prompt="test prompt", model="haiku")
        args = mock_run.call_args.args[0]
        kwargs = mock_run.call_args.kwargs
        assert args[0] == "claude"
        assert "-p" in args
        # User prompt MUST go via stdin (input_text kwarg), not argv —
        # preventing exposure through process listings on shared hosts.
        assert "test prompt" not in args
        assert kwargs.get("input_text") == "test prompt"
        assert "--system-prompt" in args
        assert SYSTEM_PROMPT in args
        assert "--tools" in args
        idx = args.index("--tools")
        assert args[idx + 1] == ""  # tools disabled
        assert "--model" in args
        assert "haiku" in args
        assert "--no-session-persistence" in args
        assert "--output-format" in args
        assert "json" in args

    @patch("mindful_harness.llm._run_subprocess_bounded")
    def test_raises_on_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed_process(
            "", returncode=1, stderr="sensitive: user@example.com sent secret"
        )
        with pytest.raises(ClaudeCLIError) as exc_info:
            _run_claude(user_prompt="test")
        # Without DEBUG, error message must NOT leak stderr content.
        assert "sensitive" not in str(exc_info.value)
        assert "exited 1" in str(exc_info.value)

    @patch("mindful_harness.llm._run_subprocess_bounded")
    def test_debug_env_var_surfaces_raw_error(
        self, mock_run: MagicMock, monkeypatch
    ) -> None:
        mock_run.return_value = _completed_process(
            "", returncode=1, stderr="sensitive context that is not a secret"
        )
        monkeypatch.setenv("MINDFUL_HARNESS_DEBUG", "1")
        with pytest.raises(ClaudeCLIError) as exc_info:
            _run_claude(user_prompt="test")
        # With DEBUG set, non-sensitive context is surfaced.
        assert "sensitive context" in str(exc_info.value)

    @patch("mindful_harness.llm._run_subprocess_bounded")
    def test_debug_mode_redacts_emails_and_keys(
        self, mock_run: MagicMock, monkeypatch
    ) -> None:
        """Even with debug enabled, emails and API-key-shaped tokens are redacted."""
        mock_run.return_value = _completed_process(
            "",
            returncode=1,
            stderr="contacted boss@example.com with api_key=sk-1234567890abcdef and token=secret-very-long",
        )
        monkeypatch.setenv("MINDFUL_HARNESS_DEBUG", "1")
        with pytest.raises(ClaudeCLIError) as exc_info:
            _run_claude(user_prompt="test")
        msg = str(exc_info.value)
        assert "boss@example.com" not in msg
        assert "<email>" in msg
        assert "sk-1234567890abcdef" not in msg
        # The api_key=... and token=... patterns should be redacted.
        assert "<key>" in msg

    @patch("mindful_harness.llm._run_subprocess_bounded")
    def test_raises_on_invalid_json(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed_process("not json")
        with pytest.raises(ClaudeCLIError, match="not valid JSON"):
            _run_claude(user_prompt="test")

    @patch("mindful_harness.llm._run_subprocess_bounded")
    def test_raises_on_is_error(self, mock_run: MagicMock) -> None:
        envelope = {"is_error": True, "result": "rate limited"}
        mock_run.return_value = _completed_process(json.dumps(envelope))
        with pytest.raises(ClaudeCLIError, match="claude reported error"):
            _run_claude(user_prompt="test")


class TestDistillItem:
    @patch("mindful_harness.llm._run_claude")
    def test_returns_structured_output_with_metadata(
        self,
        mock_run: MagicMock,
        sample_item: FirehoseItem,
        empty_mind: Mind,
        well_formed_distillation: dict,
    ) -> None:
        mock_run.return_value = _envelope(well_formed_distillation)
        result = distill_item(sample_item, empty_mind)
        assert result["framework"] == "test framework"
        assert result["_model"] == "haiku"
        assert result["_cost_usd"] == 0.01
        assert result["_duration_ms"] == 1234

    @patch("mindful_harness.llm._run_claude")
    def test_falls_back_to_result_field(
        self,
        mock_run: MagicMock,
        sample_item: FirehoseItem,
        empty_mind: Mind,
        well_formed_distillation: dict,
    ) -> None:
        envelope_no_structured = {
            "is_error": False,
            "result": "```json\n" + json.dumps(well_formed_distillation) + "\n```",
            "total_cost_usd": 0.01,
            "duration_ms": 1234,
        }
        mock_run.return_value = envelope_no_structured
        result = distill_item(sample_item, empty_mind)
        assert result["framework"] == "test framework"


class TestApplyDistillation:
    def test_mutates_mind(
        self,
        sample_item: FirehoseItem,
        empty_mind: Mind,
        well_formed_distillation: dict,
    ) -> None:
        apply_distillation(well_formed_distillation, empty_mind, sample_item)
        assert len(empty_mind.ingestion_log) == 1
        assert "test_belief" in empty_mind.beliefs
        assert empty_mind.beliefs["test_belief"].value == "could be A"
        assert empty_mind.beliefs["test_belief"].framing == "test framework"
        assert len(empty_mind.distinction_log) == 1
        assert len(empty_mind.questions) == 1
        assert len(empty_mind.interests) == 1
        assert len(empty_mind.opportunities) == 1
        assert len(empty_mind.ideas) == 1

    def test_skips_malformed_belief_silently(
        self, sample_item: FirehoseItem, empty_mind: Mind
    ) -> None:
        bad_distillation = {
            "framework": "f",
            "not_looking_for": "x",
            "belief_updates": [
                {
                    "key": "bad",
                    "value": "V",
                    "confidence": 0.5,
                    "alternatives": ["only-one"],  # too few for Conditional
                }
            ],
        }
        apply_distillation(bad_distillation, empty_mind, sample_item)
        assert "bad" not in empty_mind.beliefs

    def test_skips_empty_keys(self, sample_item: FirehoseItem, empty_mind: Mind) -> None:
        d = {
            "framework": "f",
            "not_looking_for": "x",
            "belief_updates": [
                {"key": "", "value": "V", "confidence": 0.5, "alternatives": ["a", "b"]},
                {"key": "good", "value": "V", "confidence": 0.5, "alternatives": ["a", "b"]},
            ],
        }
        apply_distillation(d, empty_mind, sample_item)
        assert "" not in empty_mind.beliefs
        assert "good" in empty_mind.beliefs

    def test_unexpected_build_error_leaves_mind_untouched(
        self, sample_item: FirehoseItem, empty_mind: Mind
    ) -> None:
        """If the build phase raises an unexpected exception, the Mind is not partially mutated."""
        # `confidence` is a non-numeric string; float() will raise ValueError.
        bad_distillation = {
            "framework": "f",
            "not_looking_for": "x",
            "belief_updates": [
                {
                    "key": "first",
                    "value": "OK",
                    "confidence": 0.5,
                    "alternatives": ["a", "b"],
                },
                {
                    "key": "second",
                    "value": "BAD",
                    "confidence": "not a number",
                    "alternatives": ["a", "b"],
                },
            ],
            "questions": ["should this appear?"],
        }
        with pytest.raises(ValueError):
            apply_distillation(bad_distillation, empty_mind, sample_item)

        # Build phase failed before commit, so NO mutation happened.
        assert empty_mind.beliefs == {}
        assert empty_mind.questions == []
        assert empty_mind.ingestion_log == []


class TestHandSchema:
    def test_required_fields(self) -> None:
        required = HAND_SCHEMA["required"]
        assert "chosen" in required
        assert "rejected_alternatives" in required
        assert "framings" in required
        assert "confidence" in required

    def test_min_rejected_alternatives(self) -> None:
        rej = HAND_SCHEMA["properties"]["rejected_alternatives"]
        assert rej["minItems"] == 2

    def test_min_framings(self) -> None:
        framings = HAND_SCHEMA["properties"]["framings"]
        assert framings["minProperties"] == 3


class TestExecuteHandLLM:
    @patch("mindful_harness.llm._run_claude")
    def test_returns_hand_result(self, mock_run: MagicMock) -> None:
        from mindful_harness import Mind, spawn_hand

        envelope = _envelope({})  # we'll set structured separately
        envelope["structured_output"] = {
            "chosen": "investigate Q1 driver before acting",
            "rejected_alternatives": ["assume seasonal", "assume noise"],
            "framings": {
                "data-quality-lens": "treat this as a measurement check",
                "product-lens": "treat this as a product-market-fit signal",
                "macro-lens": "treat this as economic-tailwind",
            },
            "confidence": 0.6,
            "reasoning": "The Q1 lift is anomalous given no marketing change.",
        }
        mock_run.return_value = envelope

        mind = Mind()
        hand = spawn_hand(
            mind,
            task="decide on Q1 investigation",
            framework="initial Q1 review",
        )

        result = execute_hand_llm(hand)
        assert result.chosen == "investigate Q1 driver before acting"
        assert len(result.framings) == 3
        assert result.confidence == 0.6
        assert "LLM execution complete" in hand.process_trail[-1]

    @patch("mindful_harness.llm._run_claude")
    def test_would_revise_if_propagates_to_hand_result(
        self, mock_run: MagicMock
    ) -> None:
        """The LLM's would_revise_if must flow into HandResult, not be dropped."""
        from mindful_harness import Mind, spawn_hand

        envelope = _envelope({})
        envelope["structured_output"] = {
            "chosen": "ship the patch",
            "rejected_alternatives": ["delay", "rollback"],
            "framings": {
                "engineering": "treat as a routine cleanup",
                "product": "treat as a customer-impact reduction",
                "ops": "treat as a stability investment",
            },
            "confidence": 0.7,
            "reasoning": "Q1 evidence supports it.",
            "would_revise_if": [
                "if customer error rate rises above 0.5%",
                "if support ticket volume reverses",
            ],
        }
        mock_run.return_value = envelope

        mind = Mind()
        hand = spawn_hand(mind, task="patch decision", framework="ship-it")
        result = execute_hand_llm(hand)

        assert result.would_revise_if == [
            "if customer error rate rises above 0.5%",
            "if support ticket volume reverses",
        ]
        # And to_decision uses those triggers as the default.
        decision = result.to_decision()
        assert "customer error rate" in decision.reversion_triggers[0]
