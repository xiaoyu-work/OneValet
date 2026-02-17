"""
OneValet LiteLLM Client - Unified LLM client powered by litellm

Supports all providers through a single client:
- OpenAI (GPT-4, GPT-4o, o1, etc.)
- Anthropic (Claude 3, Claude 3.5, Claude 4)
- Azure OpenAI
- Google Gemini
- Ollama (local models)
- DashScope (Qwen, Deepseek via OpenAI-compatible mode)
"""

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
    StreamChunk,
    ToolCall,
    Usage,
    StopReason,
)

logger = logging.getLogger(__name__)

# Provider -> default environment variable for API key
_PROVIDER_ENV_VARS: Dict[str, Optional[str]] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "ollama": None,
}


def build_litellm_model_string(provider: str, model: str) -> str:
    """Map provider + model to the litellm model string.

    litellm uses prefixed model strings to route to the correct provider.
    See https://docs.litellm.ai/docs/providers

    Args:
        provider: Provider name (openai, anthropic, azure, gemini, ollama, dashscope).
        model: Raw model name (e.g. "gpt-4o", "claude-3-5-sonnet-20241022").

    Returns:
        litellm-compatible model string.
    """
    provider = provider.lower()
    if provider == "openai":
        return model  # no prefix needed
    if provider == "anthropic":
        return f"anthropic/{model}"
    if provider == "azure":
        return f"azure/{model}"
    if provider == "gemini":
        return f"gemini/{model}"
    if provider == "ollama":
        return f"ollama/{model}"
    if provider == "dashscope":
        # DashScope uses OpenAI-compatible mode via base_url
        return f"openai/{model}"
    # Fallback: pass through as-is
    return model


class LiteLLMClient(BaseLLMClient):
    """
    Unified LLM client powered by litellm.

    Replaces per-provider clients (OpenAIClient, AnthropicClient, etc.)
    with a single implementation that delegates to litellm.

    Example:
        from onevalet.llm import LiteLLMClient, LLMConfig

        config = LLMConfig(model="gpt-4o", api_key="sk-xxx")
        client = LiteLLMClient(config=config, provider_name="openai")
        response = await client.chat_completion([
            {"role": "user", "content": "Hello!"}
        ])
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        provider_name: str = "openai",
        **kwargs,
    ):
        """
        Initialize LiteLLMClient.

        Args:
            config: LLMConfig instance.
            provider_name: Provider name (openai, anthropic, azure, gemini, ollama, dashscope).
            **kwargs: Overrides forwarded to BaseLLMClient / LLMConfig.
        """
        if config is None:
            if "model" not in kwargs:
                raise ValueError("model is required")
            model = kwargs.pop("model")
            config = LLMConfig(model=model, **kwargs)

        super().__init__(config, **kwargs)

        self.provider = provider_name.lower()
        self._litellm_model = build_litellm_model_string(self.provider, self.config.model)

        # Resolve API key: explicit config > env var
        api_key = self.config.api_key
        if not api_key:
            env_var = _PROVIDER_ENV_VARS.get(self.provider)
            if env_var:
                api_key = os.environ.get(env_var)
        self._api_key = api_key

        # Base kwargs shared by _call_api and _stream_api
        self._base_kwargs: Dict[str, Any] = {}
        if self.config.base_url:
            self._base_kwargs["api_base"] = self.config.base_url
        if api_key:
            self._base_kwargs["api_key"] = api_key

        logger.info(
            f"LiteLLMClient initialized: provider={self.provider}, "
            f"litellm_model={self._litellm_model}"
        )

    # ------------------------------------------------------------------
    # Core API methods
    # ------------------------------------------------------------------

    async def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Make a non-streaming call via litellm.acompletion."""
        import litellm

        # Handle media (images) - use base class helper
        media = kwargs.pop("media", None)
        if media and messages:
            messages = self._add_media_to_messages_openai(messages, media)

        model = kwargs.get("model") or self._litellm_model
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            **self._model_params(self.config.model, **kwargs),
            **self._base_kwargs,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]

        logger.info(
            f"[LiteLLM] model={model}, tools={len(tools) if tools else 0}, "
            f"messages={len(messages)}"
        )

        response = await litellm.acompletion(**params)

        # Parse response (ModelResponse follows OpenAI format)
        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                arguments = tc.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
                )

        stop_reason = self._parse_stop_reason(choice.finish_reason)

        # Parse usage
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        # Cost tracking via litellm
        cost = None
        if self.config.track_costs:
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                pass
        if usage and cost is not None:
            usage.cost = cost

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=getattr(response, "model", self.config.model),
            raw_response=response,
        )

    async def _stream_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Make a streaming call via litellm.acompletion(stream=True)."""
        import litellm

        model = kwargs.get("model") or self._litellm_model
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._model_params(self.config.model, **kwargs),
            **self._base_kwargs,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]

        response = await litellm.acompletion(**params)

        # Track tool call deltas across chunks
        tool_call_deltas: Dict[int, Dict[str, Any]] = {}

        async for chunk in response:
            if not chunk.choices:
                # Final chunk may carry only usage
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

            content = delta.content or ""

            # Accumulate tool call deltas
            tool_calls = None
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_deltas:
                        tool_call_deltas[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_call_deltas[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_deltas[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_deltas[idx]["arguments"] += tc_delta.function.arguments

            is_final = choice.finish_reason is not None
            stop_reason = None
            if is_final:
                stop_reason = self._parse_stop_reason(choice.finish_reason)
                if tool_call_deltas:
                    tool_calls = []
                    for idx in sorted(tool_call_deltas.keys()):
                        tc = tool_call_deltas[idx]
                        try:
                            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append(
                            ToolCall(id=tc["id"], name=tc["name"], arguments=args)
                        )

            yield StreamChunk(
                content=content,
                tool_calls=tool_calls,
                is_final=is_final,
                stop_reason=stop_reason,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_stop_reason(finish_reason: Optional[str]) -> StopReason:
        """Map litellm/OpenAI finish_reason to StopReason enum."""
        if finish_reason is None:
            return StopReason.END_TURN
        mapping = {
            "stop": StopReason.END_TURN,
            "end_turn": StopReason.END_TURN,
            "length": StopReason.MAX_TOKENS,
            "max_tokens": StopReason.MAX_TOKENS,
            "tool_calls": StopReason.TOOL_USE,
            "tool_use": StopReason.TOOL_USE,
            "function_call": StopReason.TOOL_USE,
            "content_filter": StopReason.CONTENT_FILTER,
            "stop_sequence": StopReason.STOP_SEQUENCE,
        }
        return mapping.get(finish_reason, StopReason.END_TURN)
