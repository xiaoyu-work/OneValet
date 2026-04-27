"""P1-7: Circuit breaker jitter + half-open probe lock."""

import pytest

from koa.llm.circuit_breaker import CircuitBreaker


def test_jitter_produces_variance():
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=10.0,
        jitter=0.5,
    )
    samples = {cb._sample_timeout() for _ in range(50)}
    # Some variance expected with jitter.
    assert len(samples) > 1
    for s in samples:
        # Within 50% bounds.
        assert 5.0 <= s <= 15.0


def test_no_jitter_is_deterministic():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, jitter=0.0)
    assert cb._sample_timeout() == 10.0


@pytest.mark.asyncio
async def test_half_open_probe_lock_prevents_thundering_herd():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, jitter=0.0)
    cb.record_failure()
    import time

    time.sleep(0.02)
    # Trigger transition OPEN -> HALF_OPEN.
    try:
        cb.check()
    except Exception:
        pass
    # Force HALF_OPEN if still OPEN (timing-dependent).
    from koa.llm.circuit_breaker import CircuitState

    if cb.state != CircuitState.HALF_OPEN:
        cb._state = CircuitState.HALF_OPEN

    lock = await cb.acquire_probe()
    assert lock is not None
    # check() now rejects because a probe is in flight.
    from koa.llm.circuit_breaker import CircuitBreakerOpenError

    with pytest.raises(CircuitBreakerOpenError):
        cb.check()
    cb.release_probe()
