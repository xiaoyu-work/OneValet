"""
FlowAgent Anthropic Client - Built-in Anthropic API client

Supports:
- Claude 3.5 Sonnet
- Claude 3 Opus, Sonnet, Haiku
- Claude 2 (legacy)
"""

import json
import os
from typing import Dict, Any, List, Optional, AsyncIterator

from .base import (
    BaseLLMClient, LLMConfig, LLMResponse, StreamChunk,
    ToolCall, Usage, StopReason
)


class AnthropicClient(BaseLLMClient):
    """
    Anthropic API client.

    Supports all Claude models with full feature support including
    tool use, vision, and streaming.

    Example:
        # Basic usage
        client = AnthropicClient(api_key="sk-ant-xxx")
        response = await client.chat_completion([
            {"role": "user", "content": "Hello!"}
        ])

        # With system prompt
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello!"}],
            system="You are a helpful assistant."
        )

        # With tools
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[weather_tool]
        )

        # Streaming
        async for chunk in client.stream_completion(messages):
            print(chunk.content, end="")
    """

    provider = "anthropic"

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        # Aliases
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize Anthropic client.

        Args:
            config: LLMConfig instance
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model name (default: claude-3-5-sonnet-20241022)
            **kwargs: Additional config options
        """
        # Get API key from env if not provided
        if config is None and "api_key" not in kwargs:
            kwargs["api_key"] = os.environ.get("ANTHROPIC_API_KEY")

        if config is None:
            if "model" not in kwargs:
                raise ValueError("model is required")
            model = kwargs.pop("model")
            config = LLMConfig(model=model, **kwargs)

        super().__init__(config, **kwargs)

    def _get_client(self):
        """Get or create the Anthropic client"""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )

            self._client = AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=self.config.default_headers or None,
            )

        return self._client

    def _format_tool(self, tool) -> Dict[str, Any]:
        """Format tool to Anthropic format"""
        if hasattr(tool, "name"):
            # ToolDefinition
            return {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
        elif isinstance(tool, dict):
            # Already formatted or OpenAI format
            if "function" in tool:
                # OpenAI format - convert
                func = tool["function"]
                return {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            return tool
        return tool

    def _add_media_to_messages(
        self,
        messages: List[Dict[str, Any]],
        media: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add media (images) to the last user message in Anthropic format.

        Anthropic uses a different format than OpenAI:
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "<base64>"
            }
        }
        """
        if not media:
            return messages

        messages = [msg.copy() for msg in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                text_content = messages[i].get("content", "")
                content_parts = []

                # Add image parts first (Anthropic prefers images before text)
                for item in media:
                    if item.get("type") == "image":
                        data = item.get("data", "")
                        media_type = item.get("media_type", "image/jpeg")

                        if data.startswith("http://") or data.startswith("https://"):
                            content_parts.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": data
                                }
                            })
                        else:
                            content_parts.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data
                                }
                            })

                # Add text part after images
                if text_content:
                    content_parts.append({
                        "type": "text",
                        "text": text_content
                    })

                messages[i]["content"] = content_parts
                break

        return messages

    async def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Make Anthropic API call"""
        client = self._get_client()

        # Handle media (images)
        media = kwargs.pop("media", None)
        if media and messages:
            messages = self._add_media_to_messages(messages, media)

        # Extract system message
        system = kwargs.get("system", "")
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        # Build request params
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        # Add system prompt if present
        if system:
            params["system"] = system

        # Add temperature (not supported for some models)
        if "temperature" in kwargs or self.config.temperature != 1.0:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)

        # Add tools if provided
        if tools:
            params["tools"] = [self._format_tool(t) for t in tools]
            params["tool_choice"] = kwargs.get("tool_choice", {"type": "auto"})

        # Add stop sequences if provided
        if "stop" in kwargs:
            params["stop_sequences"] = kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]

        # Make the call
        response = await client.messages.create(**params)

        # Parse response
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        # Parse stop reason
        stop_reason = self._parse_stop_reason(response.stop_reason)

        # Parse usage
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
            usage=usage,
            model=response.model,
            raw_response=response,
        )

    async def _stream_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Make streaming Anthropic API call"""
        client = self._get_client()

        # Extract system message
        system = kwargs.get("system", "")
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        # Build request params
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if system:
            params["system"] = system

        if "temperature" in kwargs or self.config.temperature != 1.0:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)

        if tools:
            params["tools"] = [self._format_tool(t) for t in tools]
            params["tool_choice"] = kwargs.get("tool_choice", {"type": "auto"})

        if "stop" in kwargs:
            params["stop_sequences"] = kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]

        # Make streaming call
        async with client.messages.stream(**params) as stream:
            # Track tool call states
            current_tool_call: Optional[Dict[str, Any]] = None
            tool_calls: List[ToolCall] = []
            usage: Optional[Usage] = None

            async for event in stream:
                if event.type == "content_block_start":
                    if hasattr(event.content_block, "type"):
                        if event.content_block.type == "tool_use":
                            current_tool_call = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "arguments": "",
                            }

                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamChunk(content=event.delta.text)
                    elif hasattr(event.delta, "partial_json"):
                        if current_tool_call:
                            current_tool_call["arguments"] += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool_call:
                        try:
                            args = json.loads(current_tool_call["arguments"]) if current_tool_call["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append(ToolCall(
                            id=current_tool_call["id"],
                            name=current_tool_call["name"],
                            arguments=args,
                        ))
                        current_tool_call = None

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        usage = Usage(
                            prompt_tokens=0,  # Will be filled from message_start
                            completion_tokens=event.usage.output_tokens,
                            total_tokens=event.usage.output_tokens,
                        )

                elif event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens
                        if usage:
                            usage.prompt_tokens = input_tokens
                            usage.total_tokens = input_tokens + usage.completion_tokens
                        else:
                            usage = Usage(prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens)

                elif event.type == "message_stop":
                    stop_reason = self._parse_stop_reason(
                        stream.current_message_snapshot.stop_reason
                        if hasattr(stream, "current_message_snapshot") else None
                    )
                    yield StreamChunk(
                        content="",
                        tool_calls=tool_calls if tool_calls else None,
                        is_final=True,
                        stop_reason=stop_reason,
                        usage=usage,
                    )

    def _parse_stop_reason(self, stop_reason: Optional[str]) -> StopReason:
        """Parse Anthropic stop_reason to StopReason"""
        if stop_reason is None:
            return StopReason.END_TURN

        mapping = {
            "end_turn": StopReason.END_TURN,
            "max_tokens": StopReason.MAX_TOKENS,
            "stop_sequence": StopReason.STOP_SEQUENCE,
            "tool_use": StopReason.TOOL_USE,
        }
        return mapping.get(stop_reason, StopReason.END_TURN)

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Anthropic's tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        client = self._get_client()
        return await client.count_tokens(text)
