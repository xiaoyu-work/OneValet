"""Token + cost budget tracking per tenant.

Budgets are enforced in two dimensions:

- **Tokens** (input + output) per window.
- **Dollar cost** per window — uses LiteLLM's cost model as the source of truth.

The tracker is eventually-consistent: we record usage after each LLM call
finishes rather than pre-reserving.  On over-budget the *next* request is
rejected so a single in-flight request can overshoot by at most one call —
acceptable for budgets on the order of thousands of tokens.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class TokenCost:
    """Usage delta to record."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class _Window:
    start: float
    tokens: int = 0
    cost: float = 0.0


@dataclass
class BudgetLimits:
    """Per-tenant limits.  ``None`` means "no limit"."""

    tokens_per_window: Optional[int] = None
    cost_per_window_usd: Optional[float] = None
    window_seconds: float = 24 * 3600.0  # daily by default


class BudgetTracker:
    """Rolling-window budget tracker.

    Each tenant has its own window.  When ``window_seconds`` elapse since
    the window started, counters reset on the next check.
    """

    def __init__(self, limits: BudgetLimits):
        self._limits = limits
        self._windows: Dict[str, _Window] = {}
        self._lock = asyncio.Lock()

    async def check(self, tenant_id: str) -> Tuple[bool, str]:
        """Return ``(allowed, reason)`` without recording anything."""
        if self._limits.tokens_per_window is None and self._limits.cost_per_window_usd is None:
            return True, ""
        async with self._lock:
            w = self._get_window(tenant_id)
            if (
                self._limits.tokens_per_window is not None
                and w.tokens >= self._limits.tokens_per_window
            ):
                return False, "token_budget_exceeded"
            if (
                self._limits.cost_per_window_usd is not None
                and w.cost >= self._limits.cost_per_window_usd
            ):
                return False, "cost_budget_exceeded"
            return True, ""

    async def record(self, tenant_id: str, usage: TokenCost) -> None:
        async with self._lock:
            w = self._get_window(tenant_id)
            w.tokens += usage.total_tokens
            w.cost += max(0.0, usage.cost_usd)

    async def snapshot(self, tenant_id: str) -> Dict[str, float]:
        async with self._lock:
            w = self._get_window(tenant_id)
            return {
                "tokens": w.tokens,
                "cost_usd": w.cost,
                "window_remaining_s": max(
                    0.0, self._limits.window_seconds - (time.monotonic() - w.start)
                ),
                "tokens_limit": self._limits.tokens_per_window or 0,
                "cost_limit_usd": self._limits.cost_per_window_usd or 0.0,
            }

    def _get_window(self, tenant_id: str) -> _Window:
        now = time.monotonic()
        w = self._windows.get(tenant_id)
        if w is None or (now - w.start) >= self._limits.window_seconds:
            w = _Window(start=now)
            self._windows[tenant_id] = w
        return w
