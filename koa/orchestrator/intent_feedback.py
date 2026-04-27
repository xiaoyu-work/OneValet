"""Intent feedback store.

Captures the signal that closes the loop between *what the classifier
guessed* and *what actually happened*.  Downstream analyses (accuracy
dashboards, active-learning labeling queues, eval-set curation) all
start here.

Outcomes recorded
-----------------

- ``completed`` — agent(s) finished without user intervention; user did
  not immediately re-ask or cancel.  Strongest positive signal.
- ``clarify`` — the classifier short-circuited to a clarification prompt.
  The follow-up message will carry ``parent_clarification_id`` so we can
  later reconstruct (classified_intent, user_correction) pairs.
- ``cancelled`` — user said "stop"/"算了" or explicitly rejected the plan.
  Strong negative signal.
- ``retried`` — user re-asked a semantically similar question within a
  short window, which usually means the first attempt misfired.
- ``error`` — execution failed for infrastructure reasons (not a
  classification error per se).

The default :class:`InMemoryIntentFeedbackStore` is a ring buffer suitable
for single-process dev / tests.  Production deployments should provide a
durable implementation (e.g. append-only file, Postgres table) so the
feedback can be aggregated across restarts and replicas.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


OUTCOME_COMPLETED = "completed"
OUTCOME_CLARIFY = "clarify"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_RETRIED = "retried"
OUTCOME_ERROR = "error"

VALID_OUTCOMES = {
    OUTCOME_COMPLETED,
    OUTCOME_CLARIFY,
    OUTCOME_CANCELLED,
    OUTCOME_RETRIED,
    OUTCOME_ERROR,
}


@dataclass
class IntentFeedbackRecord:
    """One classification + outcome pair."""

    id: str
    tenant_id: str
    user_message: str
    intent_type: str
    domains: List[str]
    confidence: float
    source: str
    outcome: str
    created_at: float
    #: Optional pointer to the parent record when this was a clarification
    #: follow-up.  Enables reconstructing (misclassified, corrected) pairs.
    parent_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class IntentFeedbackStore(Protocol):
    async def record(
        self,
        *,
        tenant_id: str,
        user_message: str,
        intent_type: str,
        domains: List[str],
        confidence: float,
        source: str,
        outcome: str,
        parent_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> IntentFeedbackRecord: ...

    async def list_recent(
        self,
        *,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[IntentFeedbackRecord]: ...

    async def accuracy_summary(
        self,
        *,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...


class InMemoryIntentFeedbackStore:
    """Process-local ring buffer implementation.

    Args:
        capacity: Max records retained in memory.  Older records are
            evicted oldest-first when the buffer is full.
    """

    def __init__(self, capacity: int = 5000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._records: deque[IntentFeedbackRecord] = deque(maxlen=capacity)
        self._lock = asyncio.Lock()

    async def record(
        self,
        *,
        tenant_id: str,
        user_message: str,
        intent_type: str,
        domains: List[str],
        confidence: float,
        source: str,
        outcome: str,
        parent_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> IntentFeedbackRecord:
        if outcome not in VALID_OUTCOMES:
            raise ValueError(
                f"Invalid outcome {outcome!r}; must be one of {sorted(VALID_OUTCOMES)}"
            )
        rec = IntentFeedbackRecord(
            id=uuid.uuid4().hex,
            tenant_id=tenant_id,
            user_message=user_message,
            intent_type=intent_type,
            domains=list(domains),
            confidence=float(confidence),
            source=source,
            outcome=outcome,
            created_at=time.time(),
            parent_id=parent_id,
            extra=dict(extra or {}),
        )
        async with self._lock:
            self._records.append(rec)
        return rec

    async def list_recent(
        self,
        *,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[IntentFeedbackRecord]:
        async with self._lock:
            items = list(self._records)
        if tenant_id:
            items = [r for r in items if r.tenant_id == tenant_id]
        return items[-limit:][::-1]

    async def accuracy_summary(
        self,
        *,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rough positive-signal ratio.

        Treats ``completed`` as positive and ``{cancelled, retried,
        error}`` as negative.  ``clarify`` is neutral — it reduced
        confidence but did not (yet) resolve good/bad.
        """
        async with self._lock:
            items = list(self._records)
        if tenant_id:
            items = [r for r in items if r.tenant_id == tenant_id]
        total = len(items)
        if total == 0:
            return {"total": 0}
        pos = sum(1 for r in items if r.outcome == OUTCOME_COMPLETED)
        neg = sum(
            1 for r in items if r.outcome in {OUTCOME_CANCELLED, OUTCOME_RETRIED, OUTCOME_ERROR}
        )
        neutral = total - pos - neg
        by_source: Dict[str, int] = {}
        for r in items:
            by_source[r.source] = by_source.get(r.source, 0) + 1
        return {
            "total": total,
            "positive": pos,
            "negative": neg,
            "neutral": neutral,
            "positive_ratio": pos / total if total else 0.0,
            "by_source": by_source,
        }
