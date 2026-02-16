"""Tests for onevalet.llm.base — BaseLLMClient pure logic methods"""

import pytest

from onevalet.llm.base import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
    StreamChunk,
    StopReason,
    ToolCall,
    Usage,
)
from onevalet.tools.models import ToolDefinition


# ── Concrete subclass for testing (abstract methods stubbed) ──


class StubLLMClient(BaseLLMClient):
    provider = "stub"

    PRICING = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4": {"input": 0.03, "output": 0.06},
    }

    async def _call_api(self, messages, tools=None, **kwargs):
        return LLMResponse(content="stub")

    async def _stream_api(self, messages, tools=None, **kwargs):
        yield StreamChunk(content="stub", is_final=True)


@pytest.fixture
def client():
    return StubLLMClient(model="gpt-4o")


@pytest.fixture
def o1_client():
    return StubLLMClient(model="o1-mini")


# =========================================================================
# _is_restricted_model
# =========================================================================


class TestIsRestrictedModel:

    def test_o1(self, client):
        assert client._is_restricted_model("o1") is True

    def test_o1_mini(self, client):
        assert client._is_restricted_model("o1-mini") is True

    def test_o3(self, client):
        assert client._is_restricted_model("o3") is True

    def test_o4_mini(self, client):
        assert client._is_restricted_model("o4-mini") is True

    def test_gpt5(self, client):
        assert client._is_restricted_model("gpt-5") is True

    def test_gpt5_turbo(self, client):
        assert client._is_restricted_model("gpt-5-turbo") is True

    def test_gpt4o_not_restricted(self, client):
        assert client._is_restricted_model("gpt-4o") is False

    def test_gpt4_not_restricted(self, client):
        assert client._is_restricted_model("gpt-4") is False

    def test_claude_not_restricted(self, client):
        assert client._is_restricted_model("claude-3-opus") is False

    def test_case_insensitive(self, client):
        assert client._is_restricted_model("O1-Mini") is True
        assert client._is_restricted_model("GPT-5") is True

    def test_uses_config_model_when_none(self, o1_client):
        assert o1_client._is_restricted_model() is True


# =========================================================================
# _model_params
# =========================================================================


class TestModelParams:

    def test_restricted_model_params(self, client):
        params = client._model_params("o1")
        assert "max_completion_tokens" in params
        assert "temperature" not in params
        assert "top_p" not in params
        assert "max_tokens" not in params

    def test_normal_model_params(self, client):
        params = client._model_params("gpt-4o")
        assert "max_tokens" in params
        assert "temperature" in params
        assert "top_p" in params
        assert "max_completion_tokens" not in params

    def test_kwargs_override(self, client):
        params = client._model_params("gpt-4o", max_tokens=2000, temperature=0.5)
        assert params["max_tokens"] == 2000
        assert params["temperature"] == 0.5

    def test_restricted_kwargs_override(self, client):
        params = client._model_params("o1", max_tokens=8000)
        assert params["max_completion_tokens"] == 8000


# =========================================================================
# _calculate_cost
# =========================================================================


class TestCalculateCost:

    def test_known_model(self, client):
        usage = Usage(prompt_tokens=1000, completion_tokens=500)
        cost = client._calculate_cost(usage, "gpt-4o")
        # (1000/1000) * 0.005 + (500/1000) * 0.015 = 0.005 + 0.0075 = 0.0125
        assert abs(cost - 0.0125) < 1e-10

    def test_unknown_model(self, client):
        usage = Usage(prompt_tokens=100, completion_tokens=50)
        assert client._calculate_cost(usage, "unknown-model") is None

    def test_zero_tokens(self, client):
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        cost = client._calculate_cost(usage, "gpt-4o")
        assert cost == 0.0

    def test_uses_config_model(self, client):
        usage = Usage(prompt_tokens=1000, completion_tokens=1000)
        cost = client._calculate_cost(usage)
        # config model is "gpt-4o"
        expected = (1000 / 1000) * 0.005 + (1000 / 1000) * 0.015
        assert abs(cost - expected) < 1e-10


# =========================================================================
# _format_tool
# =========================================================================


class TestFormatTool:

    def test_formats_tool_definition(self, client):
        tool = ToolDefinition(
            name="search_web",
            description="Search the internet",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            executor=lambda: None,
        )
        schema = client._format_tool(tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_web"
        assert schema["function"]["description"] == "Search the internet"
        assert "query" in schema["function"]["parameters"]["properties"]


# =========================================================================
# _add_media_to_messages_openai
# =========================================================================


class TestAddMediaToMessages:

    def test_no_media_returns_original(self, client):
        msgs = [{"role": "user", "content": "hello"}]
        result = client._add_media_to_messages_openai(msgs, [])
        assert result == msgs

    def test_url_image(self, client):
        msgs = [{"role": "user", "content": "describe this"}]
        media = [{"type": "image", "data": "https://example.com/img.jpg"}]
        result = client._add_media_to_messages_openai(msgs, media)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "describe this"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/img.jpg"

    def test_base64_image(self, client):
        msgs = [{"role": "user", "content": "describe"}]
        media = [{"type": "image", "data": "abc123base64data", "media_type": "image/png"}]
        result = client._add_media_to_messages_openai(msgs, media)
        content = result[0]["content"]
        assert content[1]["image_url"]["url"] == "data:image/png;base64,abc123base64data"

    def test_targets_last_user_message(self, client):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        media = [{"type": "image", "data": "https://example.com/img.jpg"}]
        result = client._add_media_to_messages_openai(msgs, media)
        assert isinstance(result[2]["content"], list)  # last user msg
        assert isinstance(result[0]["content"], str)    # first user msg unchanged

    def test_no_user_message_unchanged(self, client):
        msgs = [{"role": "system", "content": "sys"}]
        media = [{"type": "image", "data": "https://example.com/img.jpg"}]
        result = client._add_media_to_messages_openai(msgs, media)
        assert result[0]["content"] == "sys"

    def test_does_not_mutate_original(self, client):
        msgs = [{"role": "user", "content": "hello"}]
        media = [{"type": "image", "data": "https://example.com/img.jpg"}]
        client._add_media_to_messages_openai(msgs, media)
        assert msgs[0]["content"] == "hello"  # original unchanged


# =========================================================================
# LLMResponse / dataclass helpers
# =========================================================================


class TestLLMResponse:

    def test_has_tool_calls_true(self):
        tc = ToolCall(id="1", name="x", arguments={})
        resp = LLMResponse(content="", tool_calls=[tc])
        assert resp.has_tool_calls is True

    def test_has_tool_calls_false_none(self):
        resp = LLMResponse(content="hello")
        assert resp.has_tool_calls is False

    def test_has_tool_calls_false_empty(self):
        resp = LLMResponse(content="hello", tool_calls=[])
        assert resp.has_tool_calls is False

    def test_to_dict(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        resp = LLMResponse(
            content="hi",
            stop_reason=StopReason.END_TURN,
            usage=usage,
            model="gpt-4o",
        )
        d = resp.to_dict()
        assert d["content"] == "hi"
        assert d["stop_reason"] == "end_turn"
        assert d["usage"]["total_tokens"] == 30
