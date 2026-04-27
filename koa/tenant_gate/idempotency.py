"""Idempotency store for tool-call deduplication.

Tool calls that produce external side effects (send_email, transfer_money,
create_calendar_event) must be safe to retry.  Without idempotency, a
client retry or a ReAct replan can produce duplicate side effects.

Design
------

An idempotency key must be **stable across retries of the same logical
operation**.  The most reliable source is a caller-supplied header
(e.g. ``X-Idempotency-Key``).  For best-effort in-session dedup we derive a
fallback key from ``(tenant_id, session_id, tool_name, normalized_args)``.

The store exposes:

- :meth:`begin` — atomically claim the key.  Returns ``(is_new, cached_result)``.
  If ``is_new`` is False the caller should return the cached result instead of
  re-executing.
- :meth:`complete` — record a successful result.
- :meth:`fail` — release the key so a subsequent call can retry.

Records carry a TTL so the store doesn't grow unboundedly.  The in-memory
implementation is suitable for single-process deployments and tests; a
Redis/Postgres implementation should be provided for multi-worker prod.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple


def make_idempotency_key(
    *,
    tenant_id: str,
    session_id: Optional[str],
    tool_name: str,
    arguments: Dict[str, Any],
    caller_key: Optional[str] = None,
) -> str:
    """Derive a stable idempotency key.

    Caller-supplied keys win.  Otherwise hash ``(tenant, session, tool,
    normalized_args)``.  Arguments are normalized via ``json.dumps(sort_keys=True)``
    so key ordering doesn't matter.
    """
    if caller_key:
        return f"caller:{tenant_id}:{caller_key}"
    try:
        norm = json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        norm = repr(sorted((arguments or {}).items()))
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    return f"auto:{tenant_id}:{session_id or '-'}:{tool_name}:{digest}"


@dataclass
class IdempotencyRecord:
    """Persisted idempotent operation state."""

    key: str
    created_at: float
    status: str  # "in_flight" | "completed" | "failed"
    result: Any = None
    expires_at: float = 0.0


class IdempotencyStore(Protocol):
    """Pluggable backend for idempotent tool-call dedup."""

    async def begin(self, key: str, ttl_seconds: float) -> Tuple[bool, Optional[IdempotencyRecord]]:  # noqa: E501
        """Claim ``key`` atomically.

        Returns ``(is_new, existing_record)``.  ``is_new=True`` means the
        caller is responsible for executing the op and calling
        :meth:`complete`/:meth:`fail` when done.  ``is_new=False`` means a
        record already exists; callers should check ``existing_record.status``
        and either return ``existing_record.result`` (completed) or wait /
        reject (in_flight, failed).
        """
        ...

    async def complete(self, key: str, result: Any) -> None: ...

    async def fail(self, key: str) -> None: ...

    async def get(self, key: str) -> Optional[IdempotencyRecord]: ...


class InMemoryIdempotencyStore:
    """Process-local store with TTL eviction.

    NOT production-safe across multiple workers; use only for single-process
    deployments or tests.  A Redis implementation should be dropped in for
    horizontally-scaled deployments.
    """

    def __init__(self, default_ttl_seconds: float = 3600.0) -> None:
        self.default_ttl = default_ttl_seconds
        self._records: Dict[str, IdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    async def begin(
        self, key: str, ttl_seconds: Optional[float] = None
    ) -> Tuple[bool, Optional[IdempotencyRecord]]:
        ttl = ttl_seconds or self.default_ttl
        now = time.monotonic()
        async with self._lock:
            self._evict_expired(now)
            existing = self._records.get(key)
            if existing is not None and existing.expires_at > now:
                return False, existing
            rec = IdempotencyRecord(
                key=key,
                created_at=now,
                status="in_flight",
                expires_at=now + ttl,
            )
            self._records[key] = rec
            return True, None

    async def complete(self, key: str, result: Any) -> None:
        async with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return
            rec.status = "completed"
            rec.result = result

    async def fail(self, key: str) -> None:
        async with self._lock:
            # Drop the record so a retry can proceed with a fresh slot.
            self._records.pop(key, None)

    async def get(self, key: str) -> Optional[IdempotencyRecord]:
        async with self._lock:
            self._evict_expired(time.monotonic())
            return self._records.get(key)

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, r in self._records.items() if r.expires_at <= now]
        for k in expired:
            del self._records[k]
