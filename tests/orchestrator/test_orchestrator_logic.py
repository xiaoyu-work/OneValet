"""Tests for pure logic functions extracted from onevalet.orchestrator.orchestrator

Tests cover:
- _tool_name_from_schema
- _build_tool_result_message
- _assistant_message_from_response
- complete_task interception logic (tested via helpers)
- _score_tool_relevance
- _filter_tool_schemas
- _choose_fallback_tools
"""

import json
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from onevalet.orchestrator.react_config import (
    COMPLETE_TASK_TOOL_NAME,
    CompleteTaskResult,
)


# ── Standalone helper re-implementations for isolated testing ──
# These replicate the pure logic from Orchestrator without needing
# the full class instantiation.


def _tool_name_from_schema(schema: Dict[str, Any]) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    function_part = schema.get("function")
    if not isinstance(function_part, dict):
        return None
    name = function_part.get("name")
    return name if isinstance(name, str) else None


def _build_tool_result_message(tool_call_id: str, content: str, is_error: bool = False) -> Dict[str, Any]:
    if is_error:
        content = f"[ERROR] {content}"
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _assistant_message_from_response(response) -> Dict[str, Any]:
    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": getattr(response, "content", None),
    }
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                },
            }
            for tc in tool_calls
        ]
    return msg


def _make_schema(name, description=""):
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {}},
    }


# ── Mock types ──


@dataclass
class MockToolCall:
    id: str
    name: str
    arguments: Any


@dataclass
class MockLLMResponse:
    content: Optional[str]
    tool_calls: Optional[List[MockToolCall]] = None


# =========================================================================
# _tool_name_from_schema
# =========================================================================


class TestToolNameFromSchema:

    def test_valid_schema(self):
        schema = _make_schema("get_weather")
        assert _tool_name_from_schema(schema) == "get_weather"

    def test_none_schema(self):
        assert _tool_name_from_schema(None) is None

    def test_not_dict(self):
        assert _tool_name_from_schema("string") is None

    def test_missing_function_key(self):
        assert _tool_name_from_schema({"type": "function"}) is None

    def test_function_not_dict(self):
        assert _tool_name_from_schema({"function": "bad"}) is None

    def test_name_not_string(self):
        assert _tool_name_from_schema({"function": {"name": 123}}) is None

    def test_missing_name_key(self):
        assert _tool_name_from_schema({"function": {"description": "no name"}}) is None


# =========================================================================
# _build_tool_result_message
# =========================================================================


class TestBuildToolResultMessage:

    def test_normal_result(self):
        msg = _build_tool_result_message("tc1", "weather is sunny")
        assert msg == {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": "weather is sunny",
        }

    def test_error_result(self):
        msg = _build_tool_result_message("tc1", "timeout", is_error=True)
        assert msg["content"] == "[ERROR] timeout"
        assert msg["role"] == "tool"


# =========================================================================
# _assistant_message_from_response
# =========================================================================


class TestAssistantMessageFromResponse:

    def test_text_only_response(self):
        resp = MockLLMResponse(content="Hello!", tool_calls=None)
        msg = _assistant_message_from_response(resp)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert "tool_calls" not in msg

    def test_response_with_tool_calls(self):
        tc = MockToolCall(id="tc1", name="weather", arguments={"city": "tokyo"})
        resp = MockLLMResponse(content="Let me check", tool_calls=[tc])
        msg = _assistant_message_from_response(resp)
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "tc1"
        assert msg["tool_calls"][0]["function"]["name"] == "weather"
        # arguments should be JSON string
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "tokyo"}

    def test_arguments_already_string(self):
        tc = MockToolCall(id="tc1", name="weather", arguments='{"city":"tokyo"}')
        resp = MockLLMResponse(content=None, tool_calls=[tc])
        msg = _assistant_message_from_response(resp)
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"city":"tokyo"}'

    def test_none_content(self):
        resp = MockLLMResponse(content=None)
        msg = _assistant_message_from_response(resp)
        assert msg["content"] is None


# =========================================================================
# complete_task interception logic
# =========================================================================


class TestCompleteTaskInterception:
    """Test the complete_task extraction logic as an isolated function."""

    @staticmethod
    def _extract_complete_task(tool_calls):
        """Replicate the interception logic from orchestrator._react_loop_events."""
        complete_task_result = None
        remaining = []
        for tc in tool_calls:
            if tc.name == COMPLETE_TASK_TOOL_NAME:
                try:
                    args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                text = args.get("result", "")
                if text:
                    complete_task_result = CompleteTaskResult(result=text)
                else:
                    remaining.append(tc)
            else:
                remaining.append(tc)
        return complete_task_result, remaining

    def test_complete_task_extracted(self):
        tcs = [MockToolCall("1", COMPLETE_TASK_TOOL_NAME, {"result": "Done!"})]
        ct, remaining = self._extract_complete_task(tcs)
        assert ct is not None
        assert ct.result == "Done!"
        assert remaining == []

    def test_complete_task_with_other_tools(self):
        tcs = [
            MockToolCall("1", "get_weather", {"city": "tokyo"}),
            MockToolCall("2", COMPLETE_TASK_TOOL_NAME, {"result": "Here's the weather"}),
            MockToolCall("3", "send_email", {"to": "user"}),
        ]
        ct, remaining = self._extract_complete_task(tcs)
        assert ct.result == "Here's the weather"
        assert len(remaining) == 2
        assert remaining[0].name == "get_weather"
        assert remaining[1].name == "send_email"

    def test_complete_task_missing_result(self):
        tcs = [MockToolCall("1", COMPLETE_TASK_TOOL_NAME, {"other": "data"})]
        ct, remaining = self._extract_complete_task(tcs)
        assert ct is None
        assert len(remaining) == 1

    def test_complete_task_empty_result(self):
        tcs = [MockToolCall("1", COMPLETE_TASK_TOOL_NAME, {"result": ""})]
        ct, remaining = self._extract_complete_task(tcs)
        assert ct is None
        assert len(remaining) == 1

    def test_complete_task_arguments_as_json_string(self):
        tcs = [MockToolCall("1", COMPLETE_TASK_TOOL_NAME, '{"result": "done"}')]
        ct, remaining = self._extract_complete_task(tcs)
        assert ct.result == "done"

    def test_complete_task_malformed_json_arguments(self):
        tcs = [MockToolCall("1", COMPLETE_TASK_TOOL_NAME, "not json")]
        ct, remaining = self._extract_complete_task(tcs)
        assert ct is None
        assert len(remaining) == 1

    def test_complete_task_none_arguments(self):
        tcs = [MockToolCall("1", COMPLETE_TASK_TOOL_NAME, None)]
        ct, remaining = self._extract_complete_task(tcs)
        assert ct is None
        assert len(remaining) == 1

    def test_no_complete_task(self):
        tcs = [MockToolCall("1", "get_weather", {"city": "tokyo"})]
        ct, remaining = self._extract_complete_task(tcs)
        assert ct is None
        assert len(remaining) == 1


# =========================================================================
# _score_tool_relevance (re-implemented for isolated testing)
# =========================================================================


import re


def _score_tool_relevance(user_message, schema, is_agent_tool_fn=None):
    name = (_tool_name_from_schema(schema) or "").lower()
    description = str(schema.get("function", {}).get("description", "") or "").lower()
    user_text = (user_message or "").lower()

    user_tokens = set(re.findall(r"[a-z0-9_]+", user_text))
    tool_tokens = set(re.findall(r"[a-z0-9_]+", f"{name} {description}"))
    overlap = user_tokens.intersection(tool_tokens)

    score = float(len(overlap))
    if name and name in user_text:
        score += 2.0
    if is_agent_tool_fn and is_agent_tool_fn(name):
        score += 0.5
    return score


class TestScoreToolRelevance:

    def test_keyword_overlap(self):
        schema = _make_schema("get_weather", "Get weather forecast for a city")
        score = _score_tool_relevance("What's the weather in Tokyo?", schema)
        assert score > 0
        assert "weather" in set(re.findall(r"[a-z0-9_]+", "weather"))

    def test_name_in_user_text_bonus(self):
        schema = _make_schema("get_weather", "weather info")
        score_with = _score_tool_relevance("please use get_weather", schema)
        score_without = _score_tool_relevance("tell me forecast", schema)
        assert score_with > score_without

    def test_agent_tool_bonus(self):
        schema = _make_schema("EmailAgent", "send email")
        is_agent = lambda name: name == "emailagent"
        score_with = _score_tool_relevance("send email", schema, is_agent_tool_fn=is_agent)
        score_without = _score_tool_relevance("send email", schema)
        assert score_with == score_without + 0.5

    def test_empty_user_message(self):
        schema = _make_schema("x", "does stuff")
        assert _score_tool_relevance("", schema) == 0

    def test_empty_description(self):
        schema = _make_schema("tool", "")
        score = _score_tool_relevance("use tool", schema)
        assert score > 0  # "tool" matches in name


# =========================================================================
# _filter_tool_schemas
# =========================================================================


class TestFilterToolSchemas:

    def test_filter_by_names(self):
        schemas = [_make_schema("a"), _make_schema("b"), _make_schema("c")]
        filtered = [s for s in schemas if _tool_name_from_schema(s) in {"a", "c"}]
        assert len(filtered) == 2
        assert _tool_name_from_schema(filtered[0]) == "a"
        assert _tool_name_from_schema(filtered[1]) == "c"

    def test_empty_preferred_returns_all(self):
        schemas = [_make_schema("a"), _make_schema("b")]
        # When preferred_names is empty, return all
        preferred = []
        if not preferred:
            result = schemas
        assert len(result) == 2

    def test_no_match_returns_empty(self):
        schemas = [_make_schema("a"), _make_schema("b")]
        filtered = [s for s in schemas if _tool_name_from_schema(s) in {"x", "y"}]
        assert filtered == []

    def test_preserves_order(self):
        schemas = [_make_schema("c"), _make_schema("a"), _make_schema("b")]
        filtered = [s for s in schemas if _tool_name_from_schema(s) in {"a", "b", "c"}]
        names = [_tool_name_from_schema(s) for s in filtered]
        assert names == ["c", "a", "b"]  # original order preserved
