"""
FlowAgent DashScope Client - Built-in Alibaba DashScope API client

Supports:
- Qwen-Max, Qwen-Plus, Qwen-Turbo
- Qwen-VL (vision)
- All Qwen models via DashScope API
"""

import json
import os
from typing import Dict, Any, List, Optional, AsyncIterator

from .base import (
    BaseLLMClient, LLMConfig, LLMResponse, StreamChunk,
    ToolCall, Usage, StopReason
)


class DashScopeClient(BaseLLMClient):
    """
    Alibaba DashScope API client for Qwen models.

    Example:
        # Basic usage
        client = DashScopeClient(api_key="sk-xxx")
        response = await client.chat_completion([
            {"role": "user", "content": "Hello!"}
        ])

        # With specific model
        client = DashScopeClient(
            api_key="sk-xxx",
            model="qwen-max"
        )

        # Streaming
        async for chunk in client.stream_completion(messages):
            print(chunk.content, end="")
    """

    provider = "dashscope"

    # Pricing per 1K tokens (as of 2024, in CNY converted to USD approx)
    PRICING = {
        "qwen-max": {"input": 0.004, "output": 0.012},
        "qwen-max-latest": {"input": 0.004, "output": 0.012},
        "qwen-plus": {"input": 0.0008, "output": 0.002},
        "qwen-plus-latest": {"input": 0.0008, "output": 0.002},
        "qwen-turbo": {"input": 0.0003, "output": 0.0006},
        "qwen-turbo-latest": {"input": 0.0003, "output": 0.0006},
    }

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize DashScope client.

        Args:
            config: LLMConfig instance
            api_key: DashScope API key (or set DASHSCOPE_API_KEY env var)
            model: Model name (default: qwen-max)
            **kwargs: Additional config options
        """
        # Get API key from env if not provided
        if config is None and "api_key" not in kwargs:
            kwargs["api_key"] = os.environ.get("DASHSCOPE_API_KEY")

        if config is None:
            if "model" not in kwargs:
                raise ValueError("model is required")
            model = kwargs.pop("model")
            config = LLMConfig(model=model, **kwargs)

        super().__init__(config, **kwargs)

    def _get_client(self):
        """Get or create the DashScope client"""
        if self._client is None:
            try:
                import dashscope
                from dashscope import Generation
            except ImportError:
                raise ImportError(
                    "dashscope package not installed. "
                    "Install with: pip install dashscope"
                )

            dashscope.api_key = self.config.api_key
            if self.config.base_url:
                dashscope.base_http_api_url = self.config.base_url

            self._client = Generation

        return self._client

    async def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Make DashScope API call"""
        import asyncio
        from dashscope import Generation

        client = self._get_client()

        # Build request params
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "result_format": "message",
        }

        # Add generation params
        gen_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Add tools if provided (OpenAI compatible format)
        if tools:
            params["tools"] = tools

        # Add stop sequences if provided
        if "stop" in kwargs:
            gen_params["stop"] = kwargs["stop"]

        # DashScope uses sync API, wrap in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: Generation.call(
                **params,
                **gen_params,
            )
        )

        # Check for errors
        if response.status_code != 200:
            raise Exception(f"DashScope API error: {response.code} - {response.message}")

        # Parse response
        output = response.output
        message = output.choices[0].message

        content = message.get("content", "")

        # Parse tool calls
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                ))

        # Parse stop reason
        stop_reason = self._parse_stop_reason(output.get("finish_reason"))

        # Parse usage
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.get("input_tokens", 0),
                completion_tokens=response.usage.get("output_tokens", 0),
                total_tokens=response.usage.get("total_tokens", 0),
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=params["model"],
            raw_response=response,
        )

    async def _stream_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Make streaming DashScope API call"""
        import asyncio
        from dashscope import Generation

        client = self._get_client()

        # Build request params
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "result_format": "message",
            "stream": True,
            "incremental_output": True,
        }

        # Add generation params
        gen_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        if tools:
            params["tools"] = tools

        if "stop" in kwargs:
            gen_params["stop"] = kwargs["stop"]

        # DashScope streaming uses generator
        loop = asyncio.get_event_loop()

        def get_stream():
            return Generation.call(
                **params,
                **gen_params,
            )

        response_gen = await loop.run_in_executor(None, get_stream)

        for response in response_gen:
            if response.status_code != 200:
                yield StreamChunk(
                    content="",
                    is_final=True,
                    stop_reason=StopReason.ERROR,
                )
                break

            output = response.output
            if not output.choices:
                continue

            message = output.choices[0].message
            content = message.get("content", "")

            # Check if finished
            finish_reason = output.get("finish_reason")
            is_final = finish_reason is not None

            # Parse usage on final chunk
            usage = None
            if is_final and response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.get("input_tokens", 0),
                    completion_tokens=response.usage.get("output_tokens", 0),
                    total_tokens=response.usage.get("total_tokens", 0),
                )

            # Parse tool calls on final chunk
            tool_calls = None
            if is_final and message.get("tool_calls"):
                tool_calls = []
                for tc in message["tool_calls"]:
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    ))

            yield StreamChunk(
                content=content,
                tool_calls=tool_calls,
                is_final=is_final,
                stop_reason=self._parse_stop_reason(finish_reason) if is_final else None,
                usage=usage,
            )

    def _parse_stop_reason(self, finish_reason: Optional[str]) -> StopReason:
        """Parse DashScope finish_reason to StopReason"""
        if finish_reason is None:
            return StopReason.END_TURN

        mapping = {
            "stop": StopReason.END_TURN,
            "length": StopReason.MAX_TOKENS,
            "tool_calls": StopReason.TOOL_USE,
        }
        return mapping.get(finish_reason, StopReason.END_TURN)
