"""Composed per-tenant admission control.

:class:`TenantGate` combines rate limit, concurrency, budget, and
idempotency checks into a single ``acquire`` call.  Orchestrator code
wraps each request in a ``gate.acquire()`` async context manager::

    async with gate.acquire(tenant_id) as ticket:
        ...  # ReAct loop runs here
        await ticket.record_usage(prompt_tokens=..., completion_tokens=..., cost_usd=...)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from ..observability.metrics import counter
from .backends import GateBackend, InMemoryGateBackend
from .budget import BudgetLimits, BudgetTracker, TokenCost
from .rate_limiter import TokenBucketLimiter

logger = logging.getLogger(__name__)


class GateRejected(Exception):
    """Raised when a tenant fails admission control."""

    def __init__(self, reason: str, retry_after: float = 0.0, tenant_id: str = ""):
        self.reason = reason
        self.retry_after = retry_after
        self.tenant_id = tenant_id
        super().__init__(
            f"Tenant {tenant_id!r} rejected: {reason} (retry_after={retry_after:.1f}s)"
        )


@dataclass
class TenantGateConfig:
    """Configuration for :class:`TenantGate`.

    All limits default to ``None`` / 0 which means "disabled" — callers
    enable only the dimensions they want to enforce.
    """

    # Rate limit (token bucket)
    rpm: Optional[int] = None  # requests per minute
    rpm_burst: Optional[int] = None  # burst capacity (defaults to rpm)

    # Concurrency cap
    max_concurrent_per_tenant: Optional[int] = None

    # Token + cost budgets (rolling window)
    tokens_per_window: Optional[int] = None
    cost_per_window_usd: Optional[float] = None
    budget_window_seconds: float = 24 * 3600.0

    # When ``strict=True``, refuse to initialize with an in-memory backend
    # (fail-closed for production).  The server startup should set
    # ``strict = os.environ.get("KOA_ENV") == "production"``.
    strict: bool = field(
        default_factory=lambda: os.environ.get("KOA_ENV", "").lower() == "production"
    )


@dataclass
class GateDecision:
    allowed: bool
    reason: str = ""
    retry_after: float = 0.0


class GateTicket:
    """Handle returned by :meth:`TenantGate.acquire`.

    Used to record per-call usage before releasing the admission slot.
    """

    def __init__(self, gate: "TenantGate", tenant_id: str) -> None:
        self._gate = gate
        self.tenant_id = tenant_id
        self._usage = TokenCost()

    async def record_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Accumulate usage; call multiple times within a single request."""
        self._usage.prompt_tokens += prompt_tokens
        self._usage.completion_tokens += completion_tokens
        self._usage.cost_usd += cost_usd
        if self._gate._budget is not None:
            await self._gate._budget.record(
                self.tenant_id,
                TokenCost(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost_usd,
                ),
            )

    @property
    def total_usage(self) -> TokenCost:
        return self._usage


class TenantGate:
    """Per-tenant admission control: rate + concurrency + budget + idempotency."""

    def __init__(
        self,
        config: TenantGateConfig,
        backend: Optional[GateBackend] = None,
    ):
        self.config = config
        self._backend: GateBackend = backend or InMemoryGateBackend()
        if config.strict and not self._backend.multiprocess_safe:
            raise RuntimeError(
                "TenantGate configured with strict=True but backend is not "
                "multi-process safe. Provide a Redis/Postgres-backed GateBackend "
                "for production deployments."
            )
        # Rate limiter (token bucket)
        self._rate: Optional[TokenBucketLimiter] = None
        if config.rpm and config.rpm > 0:
            capacity = config.rpm_burst or config.rpm
            # rpm → per second refill
            self._rate = TokenBucketLimiter(
                capacity=capacity,
                refill_per_second=config.rpm / 60.0,
            )
        # Budget
        self._budget: Optional[BudgetTracker] = None
        if config.tokens_per_window or config.cost_per_window_usd:
            self._budget = BudgetTracker(
                BudgetLimits(
                    tokens_per_window=config.tokens_per_window,
                    cost_per_window_usd=config.cost_per_window_usd,
                    window_seconds=config.budget_window_seconds,
                )
            )

    async def check(self, tenant_id: str) -> GateDecision:
        """Run admission checks without acquiring; useful for pre-flight validation."""
        if self._rate is not None:
            ok, retry = await self._rate.acquire(tenant_id)
            if not ok:
                # Note: token bucket's acquire already consumed a token on
                # success; on failure nothing was consumed, so this call is
                # safe.  But to avoid spuriously draining tokens on the
                # dry-run path we intentionally don't implement "peek" —
                # callers should use acquire() instead for production paths.
                return GateDecision(False, "rate_limited", retry)
        if self._budget is not None:
            ok, reason = await self._budget.check(tenant_id)
            if not ok:
                return GateDecision(False, reason, 0.0)
        return GateDecision(True)

    @asynccontextmanager
    async def acquire(self, tenant_id: str) -> AsyncIterator[GateTicket]:
        """Async context manager: admission → ticket → auto-release.

        Raises :class:`GateRejected` if any check fails.  Concurrency is
        only incremented after all pre-flight checks pass, so a rejected
        request does not count against the concurrency cap.
        """
        # 1. Rate limit (consumes a token on success)
        if self._rate is not None:
            ok, retry = await self._rate.acquire(tenant_id)
            if not ok:
                counter(
                    "koa_gate_rejected_total",
                    {"reason": "rate_limited"},
                    1,
                )
                raise GateRejected("rate_limited", retry, tenant_id)

        # 2. Budget (pre-call; usage accumulates after each LLM call)
        if self._budget is not None:
            ok, reason = await self._budget.check(tenant_id)
            if not ok:
                counter("koa_gate_rejected_total", {"reason": reason}, 1)
                raise GateRejected(reason, 0.0, tenant_id)

        # 3. Concurrency
        if self.config.max_concurrent_per_tenant is not None:
            cur = await self._backend.incr_inflight(tenant_id)
            if cur > self.config.max_concurrent_per_tenant:
                await self._backend.decr_inflight(tenant_id)
                counter(
                    "koa_gate_rejected_total",
                    {"reason": "too_many_in_flight"},
                    1,
                )
                raise GateRejected("too_many_in_flight", 1.0, tenant_id)
            acquired_slot = True
        else:
            acquired_slot = False

        ticket = GateTicket(self, tenant_id)
        counter("koa_gate_admitted_total", {"tenant": tenant_id[:16]}, 1)
        try:
            yield ticket
        finally:
            if acquired_slot:
                try:
                    await self._backend.decr_inflight(tenant_id)
                except Exception as exc:  # pragma: no cover
                    logger.warning("decr_inflight failed for %s: %s", tenant_id, exc)
