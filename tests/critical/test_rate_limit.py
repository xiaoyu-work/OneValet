"""P0-2/P1-9: Tenant rate limiting."""

import asyncio

import pytest

from koa.tenant_gate.rate_limiter import SlidingWindowLimiter, TokenBucketLimiter


@pytest.mark.asyncio
async def test_token_bucket_blocks_over_capacity():
    lim = TokenBucketLimiter(capacity=3, refill_per_second=0.001)
    assert (await lim.acquire("t"))[0]
    assert (await lim.acquire("t"))[0]
    assert (await lim.acquire("t"))[0]
    allowed, retry = await lim.acquire("t")
    assert not allowed
    assert retry > 0


@pytest.mark.asyncio
async def test_token_bucket_refills():
    lim = TokenBucketLimiter(capacity=2, refill_per_second=100.0)
    await lim.acquire("t")
    await lim.acquire("t")
    allowed, _ = await lim.acquire("t")
    assert not allowed
    await asyncio.sleep(0.05)
    allowed, _ = await lim.acquire("t")
    assert allowed


@pytest.mark.asyncio
async def test_sliding_window_enforces_max():
    lim = SlidingWindowLimiter(max_requests=2, window_seconds=60)
    assert (await lim.acquire("a"))[0]
    assert (await lim.acquire("a"))[0]
    allowed, _ = await lim.acquire("a")
    assert not allowed
    # Different tenant unaffected
    assert (await lim.acquire("b"))[0]
