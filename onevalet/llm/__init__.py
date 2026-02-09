"""
OneValet LLM Client - Built-in LLM client implementations

Provides ready-to-use LLM clients for:
- OpenAI (GPT-4, GPT-3.5, o1)
- Anthropic (Claude 3, Claude 3.5)
- DashScope (Qwen)
- Google Gemini
- Deepseek
- Ollama (local models)
- OpenAI-compatible APIs

Usage:
    from onevalet.llm import OpenAIClient, AnthropicClient

    # OpenAI
    client = OpenAIClient(api_key="sk-xxx")
    response = await client.chat_completion(messages=[...])

    # With streaming
    async for chunk in client.stream_completion(messages=[...]):
        print(chunk.content)

    # Anthropic
    client = AnthropicClient(api_key="sk-ant-xxx")
    response = await client.chat_completion(messages=[...])
"""

from .base import BaseLLMClient, LLMConfig, LLMResponse, StreamChunk
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .azure_client import AzureOpenAIClient
from .dashscope_client import DashScopeClient
from .gemini_client import GeminiClient
from .ollama_client import OllamaClient
from .registry import LLMRegistry, LLMProviderConfig

__all__ = [
    # Base
    "BaseLLMClient",
    "LLMConfig",
    "LLMResponse",
    "StreamChunk",
    # Registry
    "LLMRegistry",
    "LLMProviderConfig",
    # Clients
    "OpenAIClient",
    "AnthropicClient",
    "AzureOpenAIClient",
    "DashScopeClient",
    "GeminiClient",
    "OllamaClient",
]
