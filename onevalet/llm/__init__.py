"""
OneValet LLM Client - Unified LLM client via litellm

Provides a single LiteLLMClient that supports all providers:
- OpenAI (GPT-4, GPT-4o, o1)
- Anthropic (Claude 3, Claude 3.5, Claude 4)
- Azure OpenAI
- Google Gemini
- Ollama (local models)
- DashScope (Qwen, Deepseek)

Usage:
    from onevalet.llm import LiteLLMClient, LLMConfig

    config = LLMConfig(model="gpt-4o", api_key="sk-xxx")
    client = LiteLLMClient(config=config, provider_name="openai")
    response = await client.chat_completion(messages=[...])

    # With streaming
    async for chunk in client.stream_completion(messages=[...]):
        print(chunk.content)
"""

from .base import BaseLLMClient, LLMConfig, LLMResponse, StreamChunk
from .litellm_client import LiteLLMClient
from .registry import LLMRegistry, LLMProviderConfig
from .router import ModelRouter, RoutingRule, RoutingDecision

__all__ = [
    "BaseLLMClient",
    "LLMConfig",
    "LLMResponse",
    "StreamChunk",
    "LiteLLMClient",
    "LLMRegistry",
    "LLMProviderConfig",
    "ModelRouter",
    "RoutingRule",
    "RoutingDecision",
]
