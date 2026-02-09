"""
OneValet Ollama Client - Built-in Ollama API client for local models

Supports:
- Any model available in Ollama
- Llama 3, Mistral, Phi, etc.
- Local and remote Ollama servers
"""

import json
import os
from typing import Dict, Any, List, Optional, AsyncIterator

from .base import (
    BaseLLMClient, LLMConfig, LLMResponse, StreamChunk,
    ToolCall, Usage, StopReason
)


class OllamaClient(BaseLLMClient):
    """
    Ollama API client for local LLM inference.

    Example:
        # Basic usage (local Ollama)
        client = OllamaClient(model="llama3")
        response = await client.chat_completion([
            {"role": "user", "content": "Hello!"}
        ])

        # Remote Ollama server
        client = OllamaClient(
            base_url="http://remote-server:11434",
            model="mistral"
        )

        # Streaming
        async for chunk in client.stream_completion(messages):
            print(chunk.content, end="")

        # List available models
        models = await client.list_models()
    """

    provider = "ollama"

    # No pricing for local models
    PRICING = {}

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize Ollama client.

        Args:
            config: LLMConfig instance
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Model name (default: llama3)
            **kwargs: Additional config options
        """
        if config is None and "base_url" not in kwargs:
            kwargs["base_url"] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

        if config is None:
            if "model" not in kwargs:
                raise ValueError("model is required")
            model = kwargs.pop("model")
            config = LLMConfig(model=model, **kwargs)

        super().__init__(config, **kwargs)

    def _get_client(self):
        """Get or create the Ollama client"""
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx package not installed. "
                    "Install with: pip install httpx"
                )

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            )

        return self._client

    async def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Make Ollama API call"""
        client = self._get_client()

        # Build request
        data = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }
        }

        # Add tools if provided (Ollama supports OpenAI-compatible tools)
        if tools:
            data["tools"] = tools

        # Make request
        response = await client.post("/api/chat", json=data)
        response.raise_for_status()
        result = response.json()

        # Parse response
        message = result.get("message", {})
        content = message.get("content", "")

        # Parse tool calls
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=tc.get("id", func.get("name", "")),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                ))

        # Parse usage (Ollama provides eval metrics)
        usage = None
        if "prompt_eval_count" in result or "eval_count" in result:
            usage = Usage(
                prompt_tokens=result.get("prompt_eval_count", 0),
                completion_tokens=result.get("eval_count", 0),
                total_tokens=result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
            )

        # Determine stop reason
        stop_reason = StopReason.END_TURN
        if result.get("done_reason") == "length":
            stop_reason = StopReason.MAX_TOKENS
        elif tool_calls:
            stop_reason = StopReason.TOOL_USE

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=data["model"],
            raw_response=result,
        )

    async def _stream_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Make streaming Ollama API call"""
        client = self._get_client()

        data = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }
        }

        if tools:
            data["tools"] = tools

        # Stream response
        async with client.stream("POST", "/api/chat", json=data) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = chunk_data.get("message", {})
                content = message.get("content", "")
                done = chunk_data.get("done", False)

                # Parse tool calls on final chunk
                tool_calls = None
                if done and message.get("tool_calls"):
                    tool_calls = []
                    for tc in message["tool_calls"]:
                        func = tc.get("function", {})
                        tool_calls.append(ToolCall(
                            id=tc.get("id", func.get("name", "")),
                            name=func.get("name", ""),
                            arguments=func.get("arguments", {}),
                        ))

                # Parse usage on final chunk
                usage = None
                if done:
                    if "prompt_eval_count" in chunk_data or "eval_count" in chunk_data:
                        usage = Usage(
                            prompt_tokens=chunk_data.get("prompt_eval_count", 0),
                            completion_tokens=chunk_data.get("eval_count", 0),
                            total_tokens=chunk_data.get("prompt_eval_count", 0) + chunk_data.get("eval_count", 0),
                        )

                stop_reason = None
                if done:
                    if chunk_data.get("done_reason") == "length":
                        stop_reason = StopReason.MAX_TOKENS
                    elif tool_calls:
                        stop_reason = StopReason.TOOL_USE
                    else:
                        stop_reason = StopReason.END_TURN

                yield StreamChunk(
                    content=content,
                    tool_calls=tool_calls,
                    is_final=done,
                    stop_reason=stop_reason,
                    usage=usage,
                )

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on the Ollama server.

        Returns:
            List of model info dicts
        """
        client = self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])

    async def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        client = self._get_client()
        response = await client.post("/api/pull", json={"name": model})
        return response.status_code == 200

    async def close(self) -> None:
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
