"""
Koa LiteLLM Client - Unified LLM client powered by litellm

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
from typing import Any, Dict, List, Optional

from .base import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
    StopReason,
    ToolCall,
    Usage,
)
from .prompt_caching import apply_anthropic_cache_control, is_anthropic_model

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
        from koa.llm import LiteLLMClient, LLMConfig

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
        api_version: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize LiteLLMClient.

        Args:
            config: LLMConfig instance.
            provider_name: Provider name (openai, anthropic, azure, gemini, ollama, dashscope).
            api_version: API version override (e.g. for Azure deployments).
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

        # Base kwargs shared across API calls
        self._base_kwargs: Dict[str, Any] = {}
        if self.config.base_url:
            self._base_kwargs["api_base"] = self.config.base_url
        if api_key:
            self._base_kwargs["api_key"] = api_key
        if api_version:
            self._base_kwargs["api_version"] = api_version

        logger.info(
            f"LiteLLMClient initialized: provider={self.provider}, "
            f"litellm_model={self._litellm_model}"
        )

        # Anthropic prompt caching: auto-enabled for Claude models.
        # Reduces input costs by ~90% on multi-turn conversations.
        self._use_prompt_caching = is_anthropic_model(self.provider, self.config.model)
        if self._use_prompt_caching:
            logger.info("Prompt caching enabled (Anthropic)")

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

        # Extended reasoning support (provider-agnostic via litellm)
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
            # Reasoning models typically require temperature=1 and no max_tokens
            params.pop("temperature", None)
            params.pop("top_p", None)
            params.pop("max_tokens", None)

        logger.info(
            f"[LiteLLM] model={model}, tools={len(tools) if tools else 0}, messages={len(messages)}"
        )

        # Apply Anthropic prompt caching: inject cache_control breakpoints
        if self._use_prompt_caching:
            params["messages"] = apply_anthropic_cache_control(messages)

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
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        stop_reason = self._parse_stop_reason(choice.finish_reason)

        # Parse usage
        usage = None
        if response.usage:
            # Cache stats (Anthropic returns these when prompt caching is active)
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0

            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )

            if self._use_prompt_caching and (cache_read or cache_creation):
                hit_pct = (
                    cache_read / response.usage.prompt_tokens * 100
                    if response.usage.prompt_tokens
                    else 0
                )
                logger.info(
                    f"[LiteLLM] Cache: {cache_read:,}/{response.usage.prompt_tokens:,} "
                    f"tokens ({hit_pct:.0f}% hit, {cache_creation:,} written)"
                )

        # Cost tracking via litellm
        cost = None
        if self.config.track_costs:
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception as e:
                logger.debug(f"Cost calculation failed: {e}")
        if usage and cost is not None:
            usage.cost = cost

        # Extract reasoning content (litellm standardizes across providers)
        thinking = getattr(message, "reasoning_content", None) or None

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=getattr(response, "model", self.config.model),
            thinking=thinking,
            raw_response=response,
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
