"""Tests for the LLM circuit breaker."""

import time
import pytest
from koa.llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState


class TestCircuitBreaker:

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_raises(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpenError):
            cb.check()

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED