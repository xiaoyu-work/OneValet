"""
Tests for FlowAgent Formatter Module

Tests cover:
- Provider detection
- Context limits
- Message formatting for different providers
- Tool formatting
- Response parsing
- Streaming chunk parsing
- Context management
"""

import pytest
from typing import List

from flowagents.formatter import (
    # Models
    Provider,
    TruncationStrategy,
    FormatterConfig,
    ToolSchema,
    FormattedRequest,
    detect_provider,
    get_context_limit,
    CONTEXT_LIMITS,
    # Formatters
    OpenAIFormatter,
    AnthropicFormatter,
    DashScopeFormatter,
    GeminiFormatter,
    get_formatter,
)
from flowagents.message import Message


# =============================================================================
# Test Models
# =============================================================================

class TestProviderDetection:
    """Tests for provider detection"""

    def test_detect_openai(self):
        """Test OpenAI model detection"""
        assert detect_provider("gpt-4") == Provider.OPENAI
        assert detect_provider("gpt-4-turbo") == Provider.OPENAI
        assert detect_provider("gpt-3.5-turbo") == Provider.OPENAI
        assert detect_provider("o1") == Provider.OPENAI
        assert detect_provider("o1-mini") == Provider.OPENAI

    def test_detect_anthropic(self):
        """Test Anthropic model detection"""
        assert detect_provider("claude-3-opus") == Provider.ANTHROPIC
        assert detect_provider("claude-3-5-sonnet") == Provider.ANTHROPIC
        assert detect_provider("claude-3-5-sonnet-20241022") == Provider.ANTHROPIC

    def test_detect_dashscope(self):
        """Test DashScope (Qwen) model detection"""
        assert detect_provider("qwen-max") == Provider.DASHSCOPE
        assert detect_provider("qwen-plus") == Provider.DASHSCOPE
        assert detect_provider("qwen-turbo") == Provider.DASHSCOPE

    def test_detect_gemini(self):
        """Test Gemini model detection"""
        assert detect_provider("gemini-pro") == Provider.GEMINI
        assert detect_provider("gemini-1.5-pro") == Provider.GEMINI

    def test_detect_deepseek(self):
        """Test Deepseek model detection"""
        assert detect_provider("deepseek-chat") == Provider.DEEPSEEK
        assert detect_provider("deepseek-coder") == Provider.DEEPSEEK

    def test_detect_unknown(self):
        """Test unknown model falls back to OpenAI-compatible"""
        assert detect_provider("unknown-model") == Provider.OPENAI_COMPATIBLE
        assert detect_provider("my-custom-model") == Provider.OPENAI_COMPATIBLE


class TestContextLimits:
    """Tests for context limit lookup"""

    def test_exact_match(self):
        """Test exact model name match"""
        assert get_context_limit("gpt-4") == 8192
        assert get_context_limit("gpt-4-turbo") == 128000
        assert get_context_limit("claude-3-5-sonnet") == 200000

    def test_partial_match(self):
        """Test partial model name match"""
        # Should match by substring
        limit = get_context_limit("gpt-4-0613")
        assert limit == 8192

    def test_unknown_model(self):
        """Test unknown model returns default"""
        assert get_context_limit("unknown-model") == CONTEXT_LIMITS["default"]


class TestFormatterConfig:
    """Tests for FormatterConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = FormatterConfig()
        assert config.model == "gpt-4"
        assert config.provider == Provider.OPENAI
        assert config.max_tokens == 4096
        assert config.temperature == 0.7

    def test_auto_detect_provider(self):
        """Test provider auto-detection"""
        config = FormatterConfig(model="claude-3-5-sonnet")
        assert config.provider == Provider.ANTHROPIC

        config = FormatterConfig(model="qwen-max")
        assert config.provider == Provider.DASHSCOPE

    def test_explicit_provider(self):
        """Test explicit provider setting"""
        config = FormatterConfig(
            provider=Provider.ANTHROPIC,
            model="my-model"
        )
        assert config.provider == Provider.ANTHROPIC

    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            "model": "gpt-4-turbo",
            "api_key": "test-key",
            "max_tokens": 8192,
            "temperature": 0.5,
        }
        config = FormatterConfig.from_dict(data)

        assert config.model == "gpt-4-turbo"
        assert config.api_key == "test-key"
        assert config.max_tokens == 8192
        assert config.temperature == 0.5

    def test_context_limit_property(self):
        """Test context_limit property"""
        config = FormatterConfig(model="gpt-4")
        assert config.context_limit == 8192

        config = FormatterConfig(model="claude-3-5-sonnet")
        assert config.context_limit == 200000


class TestToolSchema:
    """Tests for ToolSchema"""

    @pytest.fixture
    def sample_tool(self):
        return ToolSchema(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        )

    def test_to_openai_format(self, sample_tool):
        """Test OpenAI format conversion"""
        result = sample_tool.to_openai_format()

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get weather for a location"
        assert "location" in result["function"]["parameters"]["properties"]

    def test_to_anthropic_format(self, sample_tool):
        """Test Anthropic format conversion"""
        result = sample_tool.to_anthropic_format()

        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather for a location"
        assert "input_schema" in result
        assert "location" in result["input_schema"]["properties"]

    def test_to_gemini_format(self, sample_tool):
        """Test Gemini format conversion"""
        result = sample_tool.to_gemini_format()

        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather for a location"
        assert "parameters" in result


# =============================================================================
# Test OpenAI Formatter
# =============================================================================

class TestOpenAIFormatter:
    """Tests for OpenAI formatter"""

    @pytest.fixture
    def formatter(self):
        return OpenAIFormatter()

    @pytest.fixture
    def messages(self):
        return [
            Message(name="system", content="You are helpful", role="system"),
            Message(name="user", content="Hello", role="user"),
            Message(name="assistant", content="Hi there!", role="assistant"),
        ]

    @pytest.fixture
    def tools(self):
        return [
            ToolSchema(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {}}
            )
        ]

    def test_format_messages(self, formatter, messages):
        """Test message formatting"""
        result = formatter.format_messages(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_format_tools(self, formatter, tools):
        """Test tool formatting"""
        result = formatter.format_tools(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"

    def test_format_tools_empty(self, formatter):
        """Test empty tools returns None"""
        result = formatter.format_tools([])
        assert result is None

    def test_parse_response_dict(self, formatter):
        """Test parsing dict response"""
        response = {
            "choices": [{
                "message": {
                    "content": "Hello!",
                    "tool_calls": None
                },
                "finish_reason": "stop"
            }]
        }

        result = formatter.parse_response(response)

        assert result["content"] == "Hello!"
        assert result["tool_calls"] is None
        assert result["finish_reason"] == "stop"

    def test_parse_response_with_tool_calls(self, formatter):
        """Test parsing response with tool calls"""
        response = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        result = formatter.parse_response(response)

        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][0]["arguments"]["location"] == "London"

    @pytest.mark.asyncio
    async def test_parse_stream_chunk_string(self, formatter):
        """Test parsing SSE string chunk"""
        chunk = 'data: {"choices": [{"delta": {"content": "Hello"}}]}'

        result = await formatter.parse_stream_chunk(chunk)

        assert result is not None
        assert result["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_parse_stream_chunk_done(self, formatter):
        """Test parsing [DONE] chunk"""
        chunk = "[DONE]"

        result = await formatter.parse_stream_chunk(chunk)

        assert result == {"done": True}

    @pytest.mark.asyncio
    async def test_format_with_context_management(self, formatter, messages, tools):
        """Test format method with context management"""
        request = await formatter.format(messages, tools)

        assert isinstance(request, FormattedRequest)
        assert request.provider == Provider.OPENAI
        assert request.messages is not None
        assert request.tools is not None


# =============================================================================
# Test Anthropic Formatter
# =============================================================================

class TestAnthropicFormatter:
    """Tests for Anthropic formatter"""

    @pytest.fixture
    def formatter(self):
        config = FormatterConfig(
            provider=Provider.ANTHROPIC,
            model="claude-3-5-sonnet"
        )
        return AnthropicFormatter(config)

    @pytest.fixture
    def messages(self):
        return [
            Message(name="system", content="You are helpful", role="system"),
            Message(name="user", content="Hello", role="user"),
            Message(name="assistant", content="Hi!", role="assistant"),
        ]

    def test_format_messages_excludes_system(self, formatter, messages):
        """Test that system messages are excluded from messages list"""
        result = formatter.format_messages(messages)

        # System should be excluded (handled separately)
        assert len(result) == 2
        assert all(m["role"] != "system" for m in result)

    def test_format_tools(self, formatter):
        """Test tool formatting to Anthropic format"""
        tools = [
            ToolSchema(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {}}
            )
        ]

        result = formatter.format_tools(tools)

        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert "input_schema" in result[0]

    def test_parse_response(self, formatter):
        """Test parsing Anthropic response"""
        response = {
            "content": [
                {"type": "text", "text": "Hello there!"}
            ],
            "stop_reason": "end_turn"
        }

        result = formatter.parse_response(response)

        assert result["content"] == "Hello there!"
        assert result["finish_reason"] == "end_turn"

    def test_parse_response_with_tool_use(self, formatter):
        """Test parsing response with tool use"""
        response = {
            "content": [
                {"type": "text", "text": "I'll check the weather."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"location": "Paris"}
                }
            ],
            "stop_reason": "tool_use"
        }

        result = formatter.parse_response(response)

        assert "weather" in result["content"]
        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_format_extracts_system_prompt(self, formatter, messages):
        """Test that system prompt is extracted"""
        tools = []
        request = await formatter.format(messages, tools)

        assert request.system_prompt == "You are helpful"


# =============================================================================
# Test DashScope Formatter
# =============================================================================

class TestDashScopeFormatter:
    """Tests for DashScope (Qwen) formatter"""

    @pytest.fixture
    def formatter(self):
        config = FormatterConfig(
            provider=Provider.DASHSCOPE,
            model="qwen-max"
        )
        return DashScopeFormatter(config)

    @pytest.fixture
    def messages(self):
        return [
            Message(name="system", content="You are helpful", role="system"),
            Message(name="user", content="Hello", role="user"),
        ]

    def test_format_messages(self, formatter, messages):
        """Test message formatting (similar to OpenAI)"""
        result = formatter.format_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_parse_response(self, formatter):
        """Test parsing DashScope response"""
        response = {
            "output": {
                "choices": [{
                    "message": {
                        "content": "Hello!",
                        "role": "assistant"
                    }
                }],
                "finish_reason": "stop"
            }
        }

        result = formatter.parse_response(response)

        assert result["content"] == "Hello!"


# =============================================================================
# Test Gemini Formatter
# =============================================================================

class TestGeminiFormatter:
    """Tests for Gemini formatter"""

    @pytest.fixture
    def formatter(self):
        config = FormatterConfig(
            provider=Provider.GEMINI,
            model="gemini-pro"
        )
        return GeminiFormatter(config)

    @pytest.fixture
    def messages(self):
        return [
            Message(name="user", content="Hello", role="user"),
            Message(name="assistant", content="Hi!", role="assistant"),
        ]

    def test_format_messages(self, formatter, messages):
        """Test Gemini message formatting"""
        result = formatter.format_messages(messages)

        assert len(result) == 2
        # Gemini uses 'user' and 'model' roles
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"
        # Uses parts array
        assert "parts" in result[0]
        assert result[0]["parts"][0]["text"] == "Hello"

    def test_format_tools(self, formatter):
        """Test Gemini tool formatting"""
        tools = [
            ToolSchema(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {}}
            )
        ]

        result = formatter.format_tools(tools)

        assert len(result) == 1
        assert "functionDeclarations" in result[0]

    def test_parse_response(self, formatter):
        """Test parsing Gemini response"""
        response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello!"}]
                },
                "finishReason": "STOP"
            }]
        }

        result = formatter.parse_response(response)

        assert result["content"] == "Hello!"


# =============================================================================
# Test get_formatter Factory
# =============================================================================

class TestGetFormatter:
    """Tests for formatter factory function"""

    def test_get_openai_formatter(self):
        """Test getting OpenAI formatter"""
        formatter = get_formatter(model="gpt-4")
        assert isinstance(formatter, OpenAIFormatter)

    def test_get_anthropic_formatter(self):
        """Test getting Anthropic formatter"""
        formatter = get_formatter(model="claude-3-5-sonnet")
        assert isinstance(formatter, AnthropicFormatter)

    def test_get_dashscope_formatter(self):
        """Test getting DashScope formatter"""
        formatter = get_formatter(model="qwen-max")
        assert isinstance(formatter, DashScopeFormatter)

    def test_get_gemini_formatter(self):
        """Test getting Gemini formatter"""
        formatter = get_formatter(model="gemini-pro")
        assert isinstance(formatter, GeminiFormatter)

    def test_get_formatter_with_config(self):
        """Test getting formatter with config"""
        config = FormatterConfig(
            provider=Provider.ANTHROPIC,
            model="claude-3-opus",
            max_tokens=8192
        )
        formatter = get_formatter(config=config)

        assert isinstance(formatter, AnthropicFormatter)
        assert formatter.config.max_tokens == 8192


# =============================================================================
# Test Context Management
# =============================================================================

class TestContextManagement:
    """Tests for context window management"""

    @pytest.fixture
    def formatter(self):
        config = FormatterConfig(
            model="gpt-4",
            context_management={
                "enabled": True,
                "strategy": TruncationStrategy.SLIDING_WINDOW.value,
                "keep_last_n": 5,
                "reserve_tokens": 1000,
            }
        )
        return OpenAIFormatter(config)

    def test_sliding_window(self, formatter):
        """Test sliding window truncation"""
        # Create many messages
        messages = [
            Message(name="system", content="System", role="system"),
        ]
        for i in range(20):
            messages.append(
                Message(name="user", content=f"Message {i}", role="user")
            )

        result = formatter._sliding_window(messages, 5)

        # Should keep system + last 5
        assert len(result) == 6
        assert result[0].role == "system"
        assert "Message 19" in result[-1].get_text()

    def test_drop_oldest(self, formatter):
        """Test drop oldest truncation"""
        messages = [
            Message(name="system", content="System", role="system"),
            Message(name="user", content="A" * 1000, role="user"),
            Message(name="user", content="B" * 1000, role="user"),
            Message(name="user", content="C" * 100, role="user"),
        ]

        # Max 500 tokens = 2000 chars
        result = formatter._drop_oldest(messages, 500)

        # Should keep system and some messages
        assert result[0].role == "system"
        assert len(result) < len(messages)

    def test_estimate_tokens(self, formatter):
        """Test token estimation"""
        # 4 chars = 1 token
        assert formatter.estimate_tokens("Hello") == 1
        assert formatter.estimate_tokens("Hello World!") == 3


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
