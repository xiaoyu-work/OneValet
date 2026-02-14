"""
OneValet LLM Fallback Client - Automatic failover across multiple LLM providers

Provides FallbackLLMClient that wraps multiple LLM clients and automatically
falls over to the next candidate when one fails, with error classification
and exponential cooldown tracking.

Usage:
    from onevalet.llm import OpenAIClient, AnthropicClient
    from onevalet.llm.fallback import FallbackLLMClient, ModelCandidate

    candidates = [
        ModelCandidate(provider="openai", model="gpt-4", client=openai_client),
        ModelCandidate(provider="anthropic", model="claude-3-opus", client=anthropic_client),
    ]
    fallback = FallbackLLMClient(candidates)
    response = await fallback.chat_completion(messages=[...])
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .base import LLMResponse
from ..protocols import LLMClientProtocol
from ..tools.models import ToolDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelCandidate:
    """A single LLM provider/model that can be tried during fallback."""
    provider: str           # e.g. "openai", "anthropic"
    model: str              # e.g. "gpt-4", "claude-3-opus"
    client: LLMClientProtocol  # existing LLMClient instance
    api_key_id: str = ""    # optional identifier for cooldown tracking


@dataclass
class FallbackAttempt:
    """Record of a single failed attempt during fallback."""
    provider: str
    model: str
    error: str
    reason: str             # "rate_limit", "auth", "billing", "timeout", "format", "unknown"
    status_code: Optional[int] = None


@dataclass
class CooldownConfig:
    """Configuration for exponential cooldown behaviour."""
    base_seconds: float = 60.0      # initial cooldown (1 minute)
    multiplier: float = 5.0         # exponential multiplier
    max_seconds: float = 3600.0     # cap at 1 hour


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AllCandidatesExhaustedError(Exception):
    """Raised when every candidate has been tried (or is in cooldown) and none succeeded."""

    def __init__(self, attempts: List[FallbackAttempt], message: str = ""):
        self.attempts = attempts
        if not message:
            summaries = [
                f"  {a.provider}/{a.model}: {a.reason} - {a.error}" for a in attempts
            ]
            message = (
                "All LLM candidates exhausted. Attempts:\n" + "\n".join(summaries)
            )
        super().__init__(message)


# ---------------------------------------------------------------------------
# FallbackLLMClient
# ---------------------------------------------------------------------------

class FallbackLLMClient:
    """
    LLM client that tries multiple candidates in order with automatic failover.

    Implements the same ``chat_completion`` interface as BaseLLMClient /
    LLMClientProtocol so it can be used as a drop-in replacement.

    Features:
      - Tries candidates sequentially until one succeeds.
      - Classifies errors (rate_limit, auth, billing, timeout, format, unknown).
      - Applies exponential cooldown per candidate on failure.
      - Resets cooldown and error count on success.
      - Raises AllCandidatesExhaustedError with full attempt history when all
        candidates fail or are in cooldown.
    """

    def __init__(
        self,
        candidates: List[ModelCandidate],
        cooldown_config: Optional[CooldownConfig] = None,
    ) -> None:
        if not candidates:
            raise ValueError("At least one ModelCandidate is required")
        self.candidates = candidates
        self._cooldown_config = cooldown_config or CooldownConfig()
        # key -> timestamp until which the candidate is in cooldown
        self._cooldowns: Dict[str, float] = {}
        # key -> consecutive error count (reset on success)
        self._error_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public interface (matches BaseLLMClient / LLMClientProtocol)
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Union[Dict[str, Any], ToolDefinition]]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a chat completion request, trying candidates in order.

        Signature mirrors ``BaseLLMClient.chat_completion`` so that
        ``FallbackLLMClient`` is a drop-in replacement.
        """
        attempts: List[FallbackAttempt] = []

        for candidate in self.candidates:
            key = self._candidate_key(candidate)

            # Skip candidates that are still cooling down
            if self._is_in_cooldown(key):
                logger.debug(
                    "Skipping %s/%s (in cooldown until %.0fs from now)",
                    candidate.provider,
                    candidate.model,
                    self._cooldowns[key] - time.monotonic(),
                )
                attempts.append(FallbackAttempt(
                    provider=candidate.provider,
                    model=candidate.model,
                    error="candidate in cooldown",
                    reason="cooldown",
                ))
                continue

            try:
                logger.debug(
                    "Trying candidate %s/%s", candidate.provider, candidate.model,
                )
                response: LLMResponse = await candidate.client.chat_completion(
                    messages=messages,
                    tools=tools,
                    config=config,
                    **kwargs,
                )
                # Success -- reset tracking for this candidate
                self._mark_success(key)
                return response

            except Exception as exc:
                reason = self._classify_error(exc)
                status_code = self._extract_status_code(exc)

                logger.warning(
                    "Candidate %s/%s failed (reason=%s): %s",
                    candidate.provider,
                    candidate.model,
                    reason,
                    exc,
                )

                attempts.append(FallbackAttempt(
                    provider=candidate.provider,
                    model=candidate.model,
                    error=str(exc),
                    reason=reason,
                    status_code=status_code,
                ))

                self._mark_cooldown(key, reason)

        raise AllCandidatesExhaustedError(attempts)

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    _RATE_LIMIT_PATTERNS = ("rate limit", "429", "too many requests", "ratelimit")
    _AUTH_PATTERNS = ("401", "403", "invalid api key", "unauthorized", "authentication")
    _BILLING_PATTERNS = ("402", "payment required", "insufficient credits", "billing", "quota exceeded")
    _TIMEOUT_PATTERNS = ("timeout", "timed out", "deadline exceeded")
    _FORMAT_PATTERNS = ("400", "invalid request", "bad request", "malformed")

    def _classify_error(self, error: Exception) -> str:
        """Classify an exception into one of the known error categories."""
        # Check both the type name and the stringified error message
        haystack = f"{type(error).__name__} {error}".lower()

        for pattern in self._RATE_LIMIT_PATTERNS:
            if pattern in haystack:
                return "rate_limit"
        for pattern in self._AUTH_PATTERNS:
            if pattern in haystack:
                return "auth"
        for pattern in self._BILLING_PATTERNS:
            if pattern in haystack:
                return "billing"
        for pattern in self._TIMEOUT_PATTERNS:
            if pattern in haystack:
                return "timeout"
        for pattern in self._FORMAT_PATTERNS:
            if pattern in haystack:
                return "format"
        return "unknown"

    @staticmethod
    def _extract_status_code(error: Exception) -> Optional[int]:
        """Try to pull an HTTP status code out of the exception."""
        for attr in ("status_code", "code", "status"):
            val = getattr(error, attr, None)
            if isinstance(val, int):
                return val
        return None

    # ------------------------------------------------------------------
    # Cooldown management
    # ------------------------------------------------------------------

    def _candidate_key(self, candidate: ModelCandidate) -> str:
        """Unique key for a candidate (provider + model + optional api_key_id)."""
        parts = [candidate.provider, candidate.model]
        if candidate.api_key_id:
            parts.append(candidate.api_key_id)
        return ":".join(parts)

    def _is_in_cooldown(self, key: str) -> bool:
        """Return True if the candidate is currently in cooldown."""
        until = self._cooldowns.get(key)
        if until is None:
            return False
        if time.monotonic() >= until:
            # Cooldown has expired -- clean up
            del self._cooldowns[key]
            return False
        return True

    def _mark_cooldown(self, key: str, reason: str) -> None:
        """Record a failure and set the exponential cooldown for *key*."""
        count = self._error_counts.get(key, 0)
        self._error_counts[key] = count + 1

        cfg = self._cooldown_config
        cooldown_secs = min(
            cfg.base_seconds * (cfg.multiplier ** count),
            cfg.max_seconds,
        )
        self._cooldowns[key] = time.monotonic() + cooldown_secs
        logger.info(
            "Candidate %s entered cooldown for %.0fs (reason=%s, error_count=%d)",
            key,
            cooldown_secs,
            reason,
            count + 1,
        )

    def _mark_success(self, key: str) -> None:
        """Reset error count and cooldown after a successful call."""
        if key in self._error_counts:
            del self._error_counts[key]
        if key in self._cooldowns:
            del self._cooldowns[key]
