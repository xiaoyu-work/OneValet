"""
OneValet Formatter Models - Data structures for prompt formatting

This module defines:
- Provider types and model mappings
- Context window limits
- Truncation strategies
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


class Provider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DASHSCOPE = "dashscope"  # Alibaba Qwen
    GEMINI = "gemini"        # Google
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI_COMPATIBLE = "openai_compatible"


class TruncationStrategy(str, Enum):
    """Strategies for handling context window overflow"""
    SMART_TRUNCATE = "smart_truncate"  # Keep system + recent + important
    SUMMARIZE = "summarize"             # Summarize old messages
    SLIDING_WINDOW = "sliding_window"   # Keep last N messages
    DROP_OLDEST = "drop_oldest"         # Simply drop oldest messages
    ERROR = "error"                     # Raise error if overflow


# Context limits by model (tokens)
CONTEXT_LIMITS: Dict[str, int] = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "o1": 128000,
    "o1-mini": 128000,
    "o1-preview": 128000,

    # Anthropic
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku": 200000,

    # DashScope (Qwen)
    "qwen-max": 8192,
    "qwen-max-longcontext": 30000,
    "qwen-plus": 131072,
    "qwen-turbo": 131072,
    "qwen2.5-72b-instruct": 131072,

    # Gemini
    "gemini-pro": 32760,
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,

    # Deepseek
    "deepseek-chat": 64000,
    "deepseek-coder": 64000,

    # Default
    "default": 4096,
}


# Model to provider mapping
MODEL_PROVIDER_MAPPING: Dict[str, Provider] = {
    # OpenAI patterns
    "gpt-": Provider.OPENAI,
    "o1": Provider.OPENAI,

    # Anthropic patterns
    "claude-": Provider.ANTHROPIC,

    # DashScope patterns
    "qwen": Provider.DASHSCOPE,

    # Gemini patterns
    "gemini": Provider.GEMINI,

    # Deepseek patterns
    "deepseek": Provider.DEEPSEEK,
}


def detect_provider(model_name: str) -> Provider:
    """
    Detect provider from model name.

    Args:
        model_name: Model identifier

    Returns:
        Detected Provider enum value
    """
    model_lower = model_name.lower()

    for pattern, provider in MODEL_PROVIDER_MAPPING.items():
        if pattern in model_lower:
            return provider

    return Provider.OPENAI_COMPATIBLE


def get_context_limit(model_name: str) -> int:
    """
    Get context window limit for a model.

    Args:
        model_name: Model identifier

    Returns:
        Context limit in tokens
    """
    # Try exact match first
    if model_name in CONTEXT_LIMITS:
        return CONTEXT_LIMITS[model_name]

    # Try partial match
    model_lower = model_name.lower()
    for key, limit in CONTEXT_LIMITS.items():
        if key in model_lower:
            return limit

    return CONTEXT_LIMITS["default"]


@dataclass
class FormatterConfig:
    """
    Configuration for prompt formatter.

    Attributes:
        provider: LLM provider
        model: Model name
        api_key: API key
        api_base: API base URL (for self-hosted)
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        context_management: Context management settings
    """
    provider: Optional[Provider] = None
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    context_management: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "strategy": TruncationStrategy.SMART_TRUNCATE.value,
        "reserve_tokens": 1000,  # Reserve for response
        "keep_system": True,
        "keep_last_n": 10,
    })

    def __post_init__(self):
        # Auto-detect provider if not specified
        if self.provider is None:
            self.provider = detect_provider(self.model)

    @property
    def context_limit(self) -> int:
        """Get context limit for this model"""
        return get_context_limit(self.model)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormatterConfig":
        """Create from dictionary"""
        provider = data.get("provider")
        if isinstance(provider, str):
            provider = Provider(provider)

        return cls(
            provider=provider,
            model=data.get("model", "gpt-4"),
            api_key=data.get("api_key"),
            api_base=data.get("api_base"),
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0.7),
            context_management=data.get("context_management", {}),
        )


@dataclass
class ToolSchema:
    """
    Normalized tool schema for formatting.

    This is the internal representation that gets converted
    to provider-specific formats.
    """
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tools format"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }

    def to_gemini_format(self) -> Dict[str, Any]:
        """Convert to Gemini function declarations format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class FormattedRequest:
    """
    Formatted request ready to send to provider.

    Contains the provider-specific format of messages and tools.
    """
    provider: Provider
    model: str
    messages: Any  # Provider-specific format
    tools: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None  # For Anthropic
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API call"""
        result = {
            "model": self.model,
            "messages": self.messages,
            **self.extra_params
        }
        if self.tools:
            result["tools"] = self.tools
        if self.system_prompt and self.provider == Provider.ANTHROPIC:
            result["system"] = self.system_prompt
        return result
