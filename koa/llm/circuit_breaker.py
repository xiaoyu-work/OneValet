"""Circuit breaker for LLM API calls.

Prevents cascading failures when an LLM provider is down by fast-failing
requests after a threshold of consecutive failures. Automatically recovers
by allowing a probe request after a cooldown period.

States:
    CLOSED   → Normal operation. Failures increment counter.
    OPEN     → Provider assumed down. All calls fast-fail.
    HALF_OPEN → After recovery_timeout, one probe call is allowed.
                Success → CLOSED. Failure → OPEN.
"""

import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted on an open circuit."""

    def __init__(self, provider: str = "", retry_after: float = 0):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open for '{provider}'. "
            f"Retry after {retry_after:.0f}s."
        )


class CircuitBreaker:
    """Per-provider circuit breaker.

    Args:
        failure_threshold: Consecutive failures before opening the circuit.
        recovery_timeout: Seconds to wait before transitioning to HALF_OPEN.
        provider_name: Label for logging.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        provider_name: str = "",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.provider_name = provider_name

        self.failure_count: int = 0
        self._state: CircuitState = CircuitState.CLOSED
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    def check(self) -> None:
        """Check if a call is allowed. Raises CircuitBreakerOpenError if not."""
        current = self.state
        if current == CircuitState.OPEN:
            retry_after = self.recovery_timeout - (time.monotonic() - self._opened_at)
            raise CircuitBreakerOpenError(
                provider=self.provider_name,
                retry_after=max(0, retry_after),
            )

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state != CircuitState.CLOSED:
            logger.info(
                f"[CircuitBreaker] {self.provider_name}: "
                f"{self.state.value} → closed (success)"
            )
        self._state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        current = self.state

        if current == CircuitState.HALF_OPEN:
            self._open("half_open probe failed")
        elif current == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open(f"{self.failure_count} consecutive failures")

    def _open(self, reason: str) -> None:
        logger.warning(
            f"[CircuitBreaker] {self.provider_name}: "
            f"opening circuit ({reason}). "
            f"Will probe again in {self.recovery_timeout}s."
        )
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()