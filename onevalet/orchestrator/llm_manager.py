"""LLM call management mixin for the Orchestrator.

Provides retry logic, fallback provider chains, and error recovery
for LLM API calls.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMManagerMixin:
    """Mixin providing LLM call retry and fallback logic.

    Expects the following attributes on ``self`` (provided by Orchestrator):
    - ``llm_client``
    - ``_react_config``
    - ``_context_manager``
    - ``_model_router``
    """

    async def _llm_call_with_retry(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any] = None,
        llm_client_override: Optional[Any] = None,
        **extra_kwargs,
    ) -> Any:
        """LLM call with error recovery and model fallback chain.

        Recovery strategy:
        - RateLimitError -> exponential backoff
        - ContextOverflowError -> three-step recovery (trim -> truncate_all -> force_trim)
        - AuthError -> raise immediately
        - TimeoutError -> retry once

        If all retries on the primary client are exhausted, tries each
        fallback provider from ``ReactLoopConfig.fallback_providers`` in order.

        Args:
            tool_choice: Override for tool_choice param ("auto", "required", "none").
                         If None, the LLM client uses its default ("auto").
            llm_client_override: Optional LLM client to use instead of
                ``self.llm_client``.  Set by the model router when
                complexity-based routing is active.
        """
        client = llm_client_override or self.llm_client
        primary_error = await self._llm_call_single_client(
            client, messages, tool_schemas, tool_choice, **extra_kwargs,
        )
        if not isinstance(primary_error, Exception):
            return primary_error  # success — it's an LLMResponse

        # Primary failed — try fallback providers
        fallback_providers = self._react_config.fallback_providers
        if fallback_providers and self._model_router:
            registry = self._model_router.registry
            for provider_name in fallback_providers:
                fallback_client = registry.get(provider_name)
                if fallback_client is None or fallback_client is client:
                    continue
                logger.warning(f"[LLM] Primary failed, trying fallback provider: {provider_name}")
                result = await self._llm_call_single_client(
                    fallback_client, messages, tool_schemas, tool_choice, **extra_kwargs,
                )
                if not isinstance(result, Exception):
                    return result
                logger.warning(f"[LLM] Fallback provider {provider_name} also failed: {result}")

        raise primary_error

    async def _llm_call_single_client(
        self,
        client: Any,
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any] = None,
        **extra_kwargs,
    ) -> Any:
        """Try a single LLM client with retries.

        Returns the LLMResponse on success, or the last Exception on failure.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self._react_config.llm_max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {"messages": messages, **extra_kwargs}
                if tool_schemas:
                    kwargs["tools"] = tool_schemas
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice
                    logger.info(f"[LLM] Sending {len(tool_schemas)} tools, tool_choice={tool_choice or 'auto'}, sample: {json.dumps(tool_schemas[0], ensure_ascii=False)[:200]}")
                else:
                    logger.info("[LLM] Sending request with NO tools")
                response = await client.chat_completion(**kwargs)
                # Debug: log what came back
                tc = getattr(response, 'tool_calls', None)
                sr = getattr(response, 'stop_reason', None)
                content_len = len(getattr(response, 'content', '') or '')
                logger.info(f"[LLM] Response: stop_reason={sr}, tool_calls={len(tc) if tc else 0}, content_len={content_len}")
                return response

            except Exception as e:
                last_error = e
                error_name = type(e).__name__.lower()

                # Auth errors: raise immediately (no fallback can help)
                if "auth" in error_name or "authentication" in error_name or "permission" in error_name:
                    raise

                # Rate limit: exponential backoff
                if "ratelimit" in error_name or "rate_limit" in error_name or "429" in str(e):
                    delay = self._react_config.llm_retry_base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue

                # Context overflow: three-step recovery
                if "context" in error_name or "overflow" in error_name or "token" in error_name or "length" in str(e).lower():
                    if attempt == 0:
                        logger.warning("Context overflow, trimming history")
                        messages = self._context_manager.trim_if_needed(messages)
                    elif attempt == 1:
                        logger.warning("Context overflow persists, truncating all tool results")
                        messages = self._context_manager.truncate_all_tool_results(messages)
                    else:
                        logger.warning("Context overflow persists, force trimming")
                        messages = self._context_manager.force_trim(messages)
                    continue

                # Timeout: retry once
                if "timeout" in error_name:
                    if attempt == 0:
                        logger.warning("LLM timeout, retrying once")
                        continue
                    break  # let fallback chain handle it

                # Unknown error: retry with backoff
                if attempt < self._react_config.llm_max_retries:
                    delay = self._react_config.llm_retry_base_delay * (2 ** attempt)
                    logger.warning(f"LLM call failed ({e}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue

                break  # exhausted retries, let fallback chain handle it

        return last_error  # type: ignore[return-value]
