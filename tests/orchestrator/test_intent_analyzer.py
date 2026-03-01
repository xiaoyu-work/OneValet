"""Tests for onevalet.orchestrator.intent_analyzer

Tests cover:
- IntentAnalyzer.analyze() with mock LLM
- _parse_result() — single intent, multi intent, domain validation, downgrade logic
- _extract_json() — clean JSON, wrapped JSON, invalid input
- _fallback() — safe defaults
"""

import json
import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from onevalet.orchestrator.intent_analyzer import (
    IntentAnalyzer,
    IntentAnalysis,
    SubTask,
    VALID_DOMAINS,
)


# ── Mock LLM Response ──


@dataclass
class MockLLMResponse:
    content: Optional[str] = None


def _make_llm_client(response_content: str) -> AsyncMock:
    """Create a mock LLM client that returns a fixed response."""
    client = AsyncMock()
    client.chat_completion.return_value = MockLLMResponse(content=response_content)
    return client


def _make_failing_llm_client(error: Exception) -> AsyncMock:
    """Create a mock LLM client that raises an exception."""
    client = AsyncMock()
    client.chat_completion.side_effect = error
    return client


# ── Tests: _extract_json ──


class TestExtractJson:
    def test_clean_json(self):
        text = '{"intent_type": "single", "domains": ["productivity"], "sub_tasks": []}'
        result = IntentAnalyzer._extract_json(text)
        assert result == {"intent_type": "single", "domains": ["productivity"], "sub_tasks": []}

    def test_json_with_markdown_fence(self):
        text = '```json\n{"intent_type": "single", "domains": ["travel"], "sub_tasks": []}\n```'
        result = IntentAnalyzer._extract_json(text)
        assert result is not None
        assert result["intent_type"] == "single"
        assert result["domains"] == ["travel"]

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"intent_type": "multi", "domains": ["a"], "sub_tasks": []}\nDone.'
        result = IntentAnalyzer._extract_json(text)
        assert result is not None
        assert result["intent_type"] == "multi"

    def test_empty_string(self):
        assert IntentAnalyzer._extract_json("") is None

    def test_none_input(self):
        assert IntentAnalyzer._extract_json(None) is None

    def test_no_json(self):
        assert IntentAnalyzer._extract_json("no json here") is None

    def test_json_array_not_object(self):
        assert IntentAnalyzer._extract_json('[1, 2, 3]') is None

    def test_malformed_json(self):
        assert IntentAnalyzer._extract_json('{"broken": }') is None


# ── Tests: _parse_result ──


class TestParseResult:
    def setup_method(self):
        self.analyzer = IntentAnalyzer(llm_client=MagicMock())

    def test_single_intent_basic(self):
        data = {"intent_type": "single", "domains": ["productivity"], "sub_tasks": []}
        result = self.analyzer._parse_result(data, "What's on my calendar?")

        assert result.intent_type == "single"
        assert result.domains == ["productivity"]
        assert result.sub_tasks == []
        assert result.raw_message == "What's on my calendar?"

    def test_multi_intent_basic(self):
        data = {
            "intent_type": "multi",
            "domains": ["communication", "productivity"],
            "sub_tasks": [
                {"id": 1, "description": "Send email", "domain": "communication", "depends_on": []},
                {"id": 2, "description": "Check calendar", "domain": "productivity", "depends_on": []},
            ],
        }
        result = self.analyzer._parse_result(data, "Send email and check calendar")

        assert result.intent_type == "multi"
        assert len(result.sub_tasks) == 2
        assert result.sub_tasks[0].id == 1
        assert result.sub_tasks[0].domain == "communication"
        assert result.sub_tasks[1].id == 2
        assert result.sub_tasks[1].depends_on == []

    def test_multi_intent_with_dependencies(self):
        data = {
            "intent_type": "multi",
            "domains": ["productivity", "communication"],
            "sub_tasks": [
                {"id": 1, "description": "Check calendar", "domain": "productivity", "depends_on": []},
                {"id": 2, "description": "Email free times", "domain": "communication", "depends_on": [1]},
            ],
        }
        result = self.analyzer._parse_result(data, "Check calendar then email Bob")

        assert result.intent_type == "multi"
        assert result.sub_tasks[1].depends_on == [1]

    def test_multi_downgrade_single_subtask(self):
        """Multi with only 1 sub-task should be downgraded to single."""
        data = {
            "intent_type": "multi",
            "domains": ["productivity"],
            "sub_tasks": [
                {"id": 1, "description": "Check calendar", "domain": "productivity"},
            ],
        }
        result = self.analyzer._parse_result(data, "Check calendar")

        assert result.intent_type == "single"
        assert result.sub_tasks == []

    def test_multi_downgrade_no_subtasks(self):
        """Multi with empty sub_tasks should be downgraded to single."""
        data = {"intent_type": "multi", "domains": ["productivity"], "sub_tasks": []}
        result = self.analyzer._parse_result(data, "Check calendar")

        assert result.intent_type == "single"
        assert result.sub_tasks == []

    def test_invalid_domains_filtered(self):
        data = {"intent_type": "single", "domains": ["invalid_domain", "productivity"], "sub_tasks": []}
        result = self.analyzer._parse_result(data, "test")

        assert result.domains == ["productivity"]

    def test_all_invalid_domains_fallback_to_all(self):
        data = {"intent_type": "single", "domains": ["fake1", "fake2"], "sub_tasks": []}
        result = self.analyzer._parse_result(data, "test")

        assert result.domains == ["all"]

    def test_missing_domains_defaults_to_all(self):
        data = {"intent_type": "single"}
        result = self.analyzer._parse_result(data, "test")

        assert result.domains == ["all"]

    def test_invalid_subtask_domain_defaults_to_general(self):
        data = {
            "intent_type": "multi",
            "domains": ["general", "productivity"],
            "sub_tasks": [
                {"id": 1, "description": "task 1", "domain": "unknown_domain"},
                {"id": 2, "description": "task 2", "domain": "productivity"},
            ],
        }
        result = self.analyzer._parse_result(data, "test")

        assert result.sub_tasks[0].domain == "general"
        assert result.sub_tasks[1].domain == "productivity"

    def test_missing_intent_type_defaults_to_single(self):
        data = {"domains": ["communication"]}
        result = self.analyzer._parse_result(data, "test")

        assert result.intent_type == "single"


# ── Tests: _fallback ──


class TestFallback:
    def setup_method(self):
        self.analyzer = IntentAnalyzer(llm_client=MagicMock())

    def test_fallback_returns_safe_defaults(self):
        result = self.analyzer._fallback("hello")

        assert result.intent_type == "single"
        assert result.domains == ["all"]
        assert result.sub_tasks == []
        assert result.raw_message == "hello"


# ── Tests: analyze (async, with mock LLM) ──


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_single_intent(self):
        response = json.dumps({
            "intent_type": "single",
            "domains": ["productivity"],
            "sub_tasks": [],
        })
        client = _make_llm_client(response)
        analyzer = IntentAnalyzer(llm_client=client)

        result = await analyzer.analyze("What's on my calendar today?")

        assert result.intent_type == "single"
        assert result.domains == ["productivity"]
        assert result.raw_message == "What's on my calendar today?"
        client.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_intent(self):
        response = json.dumps({
            "intent_type": "multi",
            "domains": ["lifestyle", "communication"],
            "sub_tasks": [
                {"id": 1, "description": "Log expense", "domain": "lifestyle", "depends_on": []},
                {"id": 2, "description": "Send slack", "domain": "communication", "depends_on": []},
            ],
        })
        client = _make_llm_client(response)
        analyzer = IntentAnalyzer(llm_client=client)

        result = await analyzer.analyze("Log lunch $15 and send a slack message")

        assert result.intent_type == "multi"
        assert len(result.sub_tasks) == 2

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        client = _make_failing_llm_client(RuntimeError("API down"))
        analyzer = IntentAnalyzer(llm_client=client)

        result = await analyzer.analyze("test message")

        assert result.intent_type == "single"
        assert result.domains == ["all"]
        assert result.raw_message == "test message"

    @pytest.mark.asyncio
    async def test_unparseable_response_falls_back(self):
        client = _make_llm_client("I'm not sure what you mean")
        analyzer = IntentAnalyzer(llm_client=client)

        result = await analyzer.analyze("test message")

        assert result.intent_type == "single"
        assert result.domains == ["all"]

    @pytest.mark.asyncio
    async def test_empty_response_falls_back(self):
        client = _make_llm_client("")
        analyzer = IntentAnalyzer(llm_client=client)

        result = await analyzer.analyze("test message")

        assert result.intent_type == "single"
        assert result.domains == ["all"]

    @pytest.mark.asyncio
    async def test_none_content_falls_back(self):
        client = AsyncMock()
        client.chat_completion.return_value = MockLLMResponse(content=None)
        analyzer = IntentAnalyzer(llm_client=client)

        result = await analyzer.analyze("test message")

        assert result.intent_type == "single"
        assert result.domains == ["all"]

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_params(self):
        response = json.dumps({"intent_type": "single", "domains": ["general"], "sub_tasks": []})
        client = _make_llm_client(response)
        analyzer = IntentAnalyzer(llm_client=client)

        await analyzer.analyze("hello")

        call_args = client.chat_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[0][0]
        config = call_args.kwargs.get("config") or call_args[1] if len(call_args) > 1 else call_args.kwargs.get("config")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hello"
        assert config["temperature"] == 0.0


# ── Tests: VALID_DOMAINS ──


class TestValidDomains:
    def test_expected_domains(self):
        assert VALID_DOMAINS == {"communication", "productivity", "lifestyle", "travel", "general"}
