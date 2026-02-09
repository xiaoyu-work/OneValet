"""
OneValet LLM Client Base - Base class and common types for LLM clients

This module provides:
- BaseLLMClient: Abstract base class for all LLM clients
- LLMConfig: Configuration dataclass
- LLMResponse: Standardized response format
- StreamChunk: Streaming chunk format
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, Any, List, Optional, Union, AsyncIterator, Callable, TypeVar
)
from enum import Enum

from ..protocols import LLMClientProtocol
from ..tools.models import ToolDefinition


class StopReason(str, Enum):
    """Reason why the LLM stopped generating"""
    END_TURN = "end_turn"           # Natural completion
    MAX_TOKENS = "max_tokens"       # Hit token limit
    STOP_SEQUENCE = "stop_sequence" # Hit stop sequence
    TOOL_USE = "tool_use"           # Model wants to use a tool
    ERROR = "error"                 # Error occurred


@dataclass
class LLMConfig:
    """
    Configuration for LLM clients.

    Attributes:
        api_key: API key for the provider
        model: Model name (e.g., "gpt-4", "claude-3-opus")
        base_url: Optional base URL override for API
        temperature: Sampling temperature (0.0 - 2.0)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries on failure
        default_headers: Additional headers to send with requests
    """
    api_key: Optional[str] = None
    model: str = "gpt-4"
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: int = 60
    max_retries: int = 3
    default_headers: Dict[str, str] = field(default_factory=dict)

    # Streaming config
    stream_timeout: int = 120

    # Cost tracking
    track_costs: bool = True

    # Extra provider-specific config (e.g., api_version for Azure)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


@dataclass
class ToolCall:
    """A tool call from the LLM"""
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class Usage:
    """Token usage information"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Cost in USD (if available)
    cost: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
        }


@dataclass
class LLMResponse:
    """
    Standardized LLM response format.

    All provider clients return this format for consistency.
    """
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    stop_reason: StopReason = StopReason.END_TURN
    usage: Optional[Usage] = None
    model: Optional[str] = None

    # Raw response for debugging
    raw_response: Optional[Any] = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls"""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls] if self.tool_calls else None,
            "stop_reason": self.stop_reason.value,
            "usage": self.usage.to_dict() if self.usage else None,
            "model": self.model,
        }


@dataclass
class StreamChunk:
    """
    A chunk from streaming response.

    Used for real-time token-by-token streaming.
    """
    content: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    is_final: bool = False
    stop_reason: Optional[StopReason] = None
    usage: Optional[Usage] = None

    # Accumulated content (all chunks so far)
    accumulated_content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls] if self.tool_calls else None,
            "is_final": self.is_final,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
        }


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All provider-specific clients inherit from this class
    and implement the abstract methods.

    Implements LLMClientProtocol for compatibility with agents.

    Example:
        class MyClient(BaseLLMClient):
            async def _call_api(self, messages, tools, config):
                # Provider-specific implementation
                pass
    """

    # Provider name (override in subclasses)
    provider: str = "unknown"

    # Cost per 1K tokens (override in subclasses for cost tracking)
    # Format: {"model_name": {"input": cost, "output": cost}}
    PRICING: Dict[str, Dict[str, float]] = {}

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize the client.

        Args:
            config: LLMConfig instance
            **kwargs: Override config values
        """
        if config is None:
            config = LLMConfig(**kwargs)
        else:
            # Apply kwargs overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config
        self._client = None  # Lazy-initialized SDK client

    def _add_media_to_messages_openai(
        self,
        messages: List[Dict[str, Any]],
        media: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add media (images) to the last user message in OpenAI vision format.

        Used by: OpenAI, Azure OpenAI, and other OpenAI-compatible providers.

        Args:
            messages: List of message dicts
            media: List of media dicts with 'type', 'data', and 'media_type'

        Returns:
            Updated messages list with images embedded
        """
        if not media:
            return messages

        messages = [msg.copy() for msg in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                text_content = messages[i].get("content", "")
                content_parts = []

                if text_content:
                    content_parts.append({"type": "text", "text": text_content})

                for item in media:
                    if item.get("type") == "image":
                        data = item.get("data", "")
                        media_type = item.get("media_type", "image/jpeg")

                        if data.startswith(("http://", "https://")):
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": data}
                            })
                        else:
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{data}"}
                            })

                messages[i]["content"] = content_parts
                break

        return messages

    @abstractmethod
    async def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Make the actual API call (provider-specific).

        Args:
            messages: List of message dicts
            tools: Optional list of tool schemas
            **kwargs: Additional provider-specific params

        Returns:
            LLMResponse with standardized format
        """
        pass

    @abstractmethod
    async def _stream_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Make streaming API call (provider-specific).

        Args:
            messages: List of message dicts
            tools: Optional list of tool schemas
            **kwargs: Additional provider-specific params

        Yields:
            StreamChunk objects
        """
        pass

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Union[Dict[str, Any], ToolDefinition]]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request.

        This is the main method for non-streaming requests.
        Implements LLMClientProtocol.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tools (dict or ToolDefinition)
            config: Optional config overrides
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content, tool_calls, usage, etc.

        Example:
            response = await client.chat_completion([
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ])
            print(response.content)
        """
        # Convert ToolDefinition to dict if needed
        tool_schemas = None
        if tools:
            tool_schemas = []
            for tool in tools:
                if isinstance(tool, ToolDefinition):
                    tool_schemas.append(self._format_tool(tool))
                else:
                    tool_schemas.append(tool)

        # Apply config overrides
        merged_kwargs = {**kwargs}
        if config:
            merged_kwargs.update(config)

        # Make the API call
        response = await self._call_api(messages, tool_schemas, **merged_kwargs)

        # Calculate cost if tracking enabled
        if self.config.track_costs and response.usage:
            response.usage.cost = self._calculate_cost(response.usage, response.model)

        return response

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Union[Dict[str, Any], ToolDefinition]]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Send a streaming chat completion request.

        Yields chunks as they arrive from the API.

        Args:
            messages: List of message dicts
            tools: Optional list of tools
            config: Optional config overrides
            **kwargs: Additional parameters

        Yields:
            StreamChunk objects with content deltas

        Example:
            async for chunk in client.stream_completion(messages):
                print(chunk.content, end="", flush=True)
                if chunk.is_final:
                    print(f"\\nUsed {chunk.usage.total_tokens} tokens")
        """
        # Convert tools
        tool_schemas = None
        if tools:
            tool_schemas = []
            for tool in tools:
                if isinstance(tool, ToolDefinition):
                    tool_schemas.append(self._format_tool(tool))
                else:
                    tool_schemas.append(tool)

        # Apply config overrides
        merged_kwargs = {**kwargs}
        if config:
            merged_kwargs.update(config)

        # Stream the response
        accumulated = ""
        async for chunk in self._stream_api(messages, tool_schemas, **merged_kwargs):
            accumulated += chunk.content
            chunk.accumulated_content = accumulated
            yield chunk

    def _format_tool(self, tool: ToolDefinition) -> Dict[str, Any]:
        """
        Format ToolDefinition to provider-specific schema.

        Default implementation uses OpenAI format.
        Override in subclasses for other formats.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        }

    def _calculate_cost(self, usage: Usage, model: Optional[str] = None) -> Optional[float]:
        """Calculate cost based on token usage"""
        model = model or self.config.model
        if model not in self.PRICING:
            return None

        pricing = self.PRICING[model]
        input_cost = (usage.prompt_tokens / 1000) * pricing.get("input", 0)
        output_cost = (usage.completion_tokens / 1000) * pricing.get("output", 0)
        return input_cost + output_cost

    async def close(self) -> None:
        """Close the client and release resources"""
        if self._client and hasattr(self._client, "close"):
            await self._client.close()
        self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
