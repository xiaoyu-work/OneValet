"""
OneValet Azure OpenAI Client - Built-in Azure OpenAI API client

Supports:
- Azure OpenAI Service deployments
- All Azure-deployed OpenAI models
"""

import json
import os
from typing import Dict, Any, List, Optional, AsyncIterator

from .base import (
    BaseLLMClient, LLMConfig, LLMResponse, StreamChunk,
    ToolCall, Usage, StopReason
)


class AzureOpenAIClient(BaseLLMClient):
    """
    Azure OpenAI API client.

    Example:
        client = AzureOpenAIClient(
            api_key="xxx",
            base_url="https://xxx.openai.azure.com/",
            model="gpt-4",
            api_version="2024-12-01-preview"
        )
        response = await client.chat_completion([
            {"role": "user", "content": "Hello!"}
        ])
    """

    provider = "azure"

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize Azure OpenAI client.

        Args:
            config: LLMConfig instance
            api_key: Azure OpenAI API key
            base_url: Azure OpenAI endpoint URL
            model: Deployment name
            api_version: API version (default: 2024-12-01-preview)
            **kwargs: Additional config options
        """
        if config is None and "api_key" not in kwargs:
            kwargs["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY")

        if config is None:
            if "model" not in kwargs:
                raise ValueError("model is required")
            model = kwargs.pop("model")
            config = LLMConfig(model=model, **kwargs)

        super().__init__(config, **kwargs)

        # Get api_version from extra or kwargs
        self.api_version = (
            kwargs.get("api_version") or
            (config.extra or {}).get("api_version") or
            "2024-12-01-preview"
        )

    def _get_client(self):
        """Get or create the Azure OpenAI client"""
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )

            self._client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.base_url,
                api_version=self.api_version,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=self.config.default_headers or None,
            )

        return self._client

    async def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Make Azure OpenAI API call"""
        client = self._get_client()

        # Handle media (images) - use base class method
        media = kwargs.pop("media", None)
        if media and messages:
            messages = self._add_media_to_messages_openai(messages, media)

        # Build request params
        model = kwargs.get("model", self.config.model)
        params = {
            "model": model,
            "messages": messages,
            **self._model_params(model, **kwargs),
        }

        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add stop sequences if provided
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]

        # Make the call
        response = await client.chat.completions.create(**params)

        # Parse response
        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        # Parse stop reason
        stop_reason = self._parse_stop_reason(choice.finish_reason)

        # Parse usage
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=response.model,
            raw_response=response.model_dump(),
        )

    async def _stream_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream Azure OpenAI API response"""
        client = self._get_client()

        # Build request params
        model = kwargs.get("model", self.config.model)
        params = {
            "model": model,
            "messages": messages,
            "stream": True,
            **self._model_params(model, **kwargs),
        }

        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Make the streaming call
        stream = await client.chat.completions.create(**params)

        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Parse content
            content = delta.content if delta.content else None

            # Parse tool calls
            tool_calls = None
            if delta.tool_calls:
                tool_calls = []
                for tc in delta.tool_calls:
                    if tc.function:
                        tool_calls.append(ToolCall(
                            id=tc.id or "",
                            name=tc.function.name or "",
                            arguments=tc.function.arguments or "",
                        ))

            # Check if done
            is_final = choice.finish_reason is not None
            stop_reason = None
            if is_final:
                stop_reason = self._parse_stop_reason(choice.finish_reason)

            yield StreamChunk(
                content=content,
                tool_calls=tool_calls,
                is_final=is_final,
                stop_reason=stop_reason,
            )

    def _parse_stop_reason(self, finish_reason: Optional[str]) -> StopReason:
        """Parse OpenAI finish reason to StopReason"""
        if finish_reason == "stop":
            return StopReason.END_TURN
        elif finish_reason == "tool_calls":
            return StopReason.TOOL_USE
        elif finish_reason == "length":
            return StopReason.MAX_TOKENS
        elif finish_reason == "content_filter":
            return StopReason.CONTENT_FILTER
        else:
            return StopReason.END_TURN
