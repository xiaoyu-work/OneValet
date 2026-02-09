"""
OneValet Formatter Module - Multi-model prompt formatting

This module provides automatic prompt formatting for different LLM providers.
The framework auto-detects the provider from model name and converts:
- Messages to provider-specific format
- Tools to provider-specific format
- Streaming responses to normalized format

Supported Providers:
- OpenAI (GPT-4, GPT-3.5, o1)
- Anthropic (Claude 3, Claude 3.5)
- DashScope (Qwen)
- Google Gemini
- Deepseek
- Ollama
- vLLM (OpenAI-compatible)

Usage:
    from onevalet.formatter import get_formatter, FormatterConfig

    # Auto-detect from model name
    formatter = get_formatter(model="claude-3-5-sonnet")

    # Or explicit provider
    formatter = get_formatter(
        config=FormatterConfig(
            provider=Provider.ANTHROPIC,
            model="claude-3-5-sonnet-20241022"
        )
    )

    # Format messages and tools
    request = await formatter.format(messages, tools)

    # Parse streaming response
    async for chunk in response:
        parsed = await formatter.parse_stream_chunk(chunk)
        if parsed and parsed.get("content"):
            print(parsed["content"], end="")

Context Management:
    The formatter automatically handles context window limits:

    config = FormatterConfig(
        model="gpt-4",
        context_management={
            "enabled": True,
            "strategy": "smart_truncate",
            "keep_last_n": 10,
            "reserve_tokens": 1000
        }
    )
"""

from .models import (
    Provider,
    TruncationStrategy,
    FormatterConfig,
    ToolSchema,
    FormattedRequest,
    detect_provider,
    get_context_limit,
    CONTEXT_LIMITS,
)

from .base import FormatterBase

from .formatters import (
    OpenAIFormatter,
    AnthropicFormatter,
    DashScopeFormatter,
    GeminiFormatter,
    DeepseekFormatter,
    OllamaFormatter,
    VLLMFormatter,
    get_formatter,
    FORMATTER_MAP,
)

__all__ = [
    # Models
    "Provider",
    "TruncationStrategy",
    "FormatterConfig",
    "ToolSchema",
    "FormattedRequest",
    "detect_provider",
    "get_context_limit",
    "CONTEXT_LIMITS",
    # Base
    "FormatterBase",
    # Formatters
    "OpenAIFormatter",
    "AnthropicFormatter",
    "DashScopeFormatter",
    "GeminiFormatter",
    "DeepseekFormatter",
    "OllamaFormatter",
    "VLLMFormatter",
    "get_formatter",
    "FORMATTER_MAP",
]
