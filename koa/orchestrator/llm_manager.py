"""LLM call management mixin for the Orchestrator.

Provides retry logic, fallback provider chains, and error recovery
for LLM API calls.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from .error_classifier import LLMErrorKind, classify_llm_error
from ..llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class LLMManagerMixin:
    """Mixin providing LLM call retry and fallback logic.

    Expects the following attributes on ``self`` (provided by Orchestrator):
    - ``llm_client``
    - ``_react_config``
    - ``_context_manager``
    - ``_model_router`` (optional — fallback still works via LLMRegistry singleton)
    """

    # Circuit breakers keyed by provider name (populated lazily)
    _circuit_breakers: dict = {}

    def _get_circuit_breaker(self, client: Any) -> CircuitBreaker:
        """Get or create a circuit breaker for the given LLM client."""
        provider = getattr(client, 'provider', '') or id(client)
        model = getattr(getattr(client, 'config', None), 'model', '')
        key = f"{provider}:{model}"
        if key not in self._circuit_breakers:
            self._circuit_breakers[key] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30.0,
                provider_name=key,
            )
        return self._circuit_breakers[key]

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
        if fallback_providers:
            registry = self._get_llm_registry()
            if registry is None:
                logger.warning(
                    "[LLM] fallback_providers configured but no LLMRegistry available; skipping fallback"
                )
                raise primary_error
            for provider_name in fallback_providers:
                fallback_client = registry.get(provider_name)
                if fallback_client is None or fallback_client is client:
                    continue
                logger.warning(f"[LLM] Primary failed, trying fallback provider: {provider_name}")
                try:
                    result = await self._llm_call_single_client(
                        fallback_client, messages, tool_schemas, tool_choice, **extra_kwargs,
                    )
                except Exception as fb_err:
                    # Auth errors raised from _llm_call_single_client — skip this fallback
                    logger.warning(f"[LLM] Fallback provider {provider_name} raised: {fb_err}")
                    continue
                if not isinstance(result, Exception):
                    logger.info(f"[LLM] Fallback to {provider_name} succeeded")
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
        cb = self._get_circuit_breaker(client)

        # Fast-fail if circuit is open
        try:
            cb.check()
        except CircuitBreakerOpenError as e:
            logger.warning(f"[LLM] Circuit open for {e.provider}, skipping retries")
            return e
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
                cb.record_success()
                return response

            except Exception as e:
                last_error = e
                error_kind = classify_llm_error(e)

                # Auth errors: raise immediately (no fallback can help)
                if error_kind == LLMErrorKind.AUTH:
                    raise

                # Rate limit: exponential backoff
                if error_kind == LLMErrorKind.RATE_LIMIT:
                    delay = self._react_config.llm_retry_base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue

                # Context overflow: three-step recovery
                if error_kind == LLMErrorKind.CONTEXT_OVERFLOW:
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
                if error_kind == LLMErrorKind.TIMEOUT:
                    if attempt == 0:
                        logger.warning("LLM timeout, retrying once")
                        continue
                    cb.record_failure()
                    break  # let fallback chain handle it

                # Bad request: don't retry (invalid params won't fix themselves)
                if error_kind == LLMErrorKind.BAD_REQUEST:
                    logger.warning(f"LLM bad request ({e}), not retrying")
                    cb.record_failure()
                    break

                # Service unavailable / transient / unknown: retry with backoff
                if attempt < self._react_config.llm_max_retries:
                    delay = self._react_config.llm_retry_base_delay * (2 ** attempt)
                    logger.warning(f"LLM call failed ({error_kind.value}: {e}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue

                cb.record_failure()
                break  # exhausted retries, let fallback chain handle it

        return last_error  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_llm_registry(self) -> Optional[Any]:
        """Return the LLMRegistry, preferring the model router's reference.

        Falls back to the singleton ``LLMRegistry.get_instance()`` so that
        fallback providers work even when ``ModelRouter`` is not configured.
        """
        if getattr(self, "_model_router", None) is not None:
            return self._model_router.registry

        try:
            from ..llm.registry import LLMRegistry
            return LLMRegistry.get_instance()
        except Exception:
            return None
