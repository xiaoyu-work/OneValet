"""
OneValet OpenAI Client - Built-in OpenAI API client

Supports:
- GPT-4, GPT-4 Turbo, GPT-4o
- GPT-3.5 Turbo
- o1, o1-mini
- Any OpenAI-compatible API (Azure, vLLM, etc.)
"""

import json
import os
from typing import Dict, Any, List, Optional, AsyncIterator

from .base import (
    BaseLLMClient, LLMConfig, LLMResponse, StreamChunk,
    ToolCall, Usage, StopReason
)


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client.

    Supports all OpenAI chat models and OpenAI-compatible APIs.

    Example:
        # Basic usage
        client = OpenAIClient(api_key="sk-xxx")
        response = await client.chat_completion([
            {"role": "user", "content": "Hello!"}
        ])

        # With tools
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[weather_tool]
        )

        # Streaming
        async for chunk in client.stream_completion(messages):
            print(chunk.content, end="")

        # Azure OpenAI
        client = OpenAIClient(
            api_key="xxx",
            base_url="https://xxx.openai.azure.com/",
            model="gpt-4"
        )
    """

    provider = "openai"

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "o1": {"input": 0.015, "output": 0.06},
        "o1-mini": {"input": 0.003, "output": 0.012},
        "o1-preview": {"input": 0.015, "output": 0.06},
    }

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize OpenAI client.

        Args:
            config: LLMConfig instance
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (default: gpt-4)
            base_url: Optional base URL for API
            **kwargs: Additional config options
        """
        # Get API key from env if not provided
        if config is None and "api_key" not in kwargs:
            kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")

        if config is None:
            if "model" not in kwargs:
                raise ValueError("model is required")
            model = kwargs.pop("model")
            config = LLMConfig(model=model, **kwargs)

        super().__init__(config, **kwargs)

    def _get_client(self):
        """Get or create the OpenAI client"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )

            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
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
        """Make OpenAI API call"""
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
            raw_response=response,
        )

    async def _stream_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Make streaming OpenAI API call"""
        client = self._get_client()

        # Build request params
        model = kwargs.get("model", self.config.model)
        params = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._model_params(model, **kwargs),
        }

        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add stop sequences if provided
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]

        # Make the streaming call
        stream = await client.chat.completions.create(**params)

        # Track tool call deltas
        tool_call_deltas: Dict[int, Dict[str, Any]] = {}

        async for chunk in stream:
            if not chunk.choices:
                # Final chunk with usage
                if chunk.usage:
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        usage=Usage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        ),
                    )
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Extract content
            content = delta.content or ""

            # Track tool calls
            tool_calls = None
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_deltas:
                        tool_call_deltas[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }

                    if tc_delta.id:
                        tool_call_deltas[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_deltas[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_deltas[idx]["arguments"] += tc_delta.function.arguments

            # Check if finished
            is_final = choice.finish_reason is not None
            stop_reason = None
            if is_final:
                stop_reason = self._parse_stop_reason(choice.finish_reason)

                # Parse completed tool calls
                if tool_call_deltas:
                    tool_calls = []
                    for idx in sorted(tool_call_deltas.keys()):
                        tc = tool_call_deltas[idx]
                        try:
                            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append(ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=args,
                        ))

            yield StreamChunk(
                content=content,
                tool_calls=tool_calls,
                is_final=is_final,
                stop_reason=stop_reason,
            )

    def _parse_stop_reason(self, finish_reason: Optional[str]) -> StopReason:
        """Parse OpenAI finish_reason to StopReason"""
        if finish_reason is None:
            return StopReason.END_TURN

        mapping = {
            "stop": StopReason.END_TURN,
            "length": StopReason.MAX_TOKENS,
            "tool_calls": StopReason.TOOL_USE,
            "content_filter": StopReason.END_TURN,
            "function_call": StopReason.TOOL_USE,  # Legacy
        }
        return mapping.get(finish_reason, StopReason.END_TURN)

    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """
        Create an embedding for text.

        Args:
            text: Text to embed
            model: Embedding model name

        Returns:
            List of floats representing the embedding
        """
        client = self._get_client()
        response = await client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding
