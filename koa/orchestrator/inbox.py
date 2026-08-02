"""The Inbox: what the assistant needs from a person, and how they answer it.

An agent that needs a decision only a human can make records an ask here and
its run stops. The person answers whenever they get to it, from whichever
surface the ask was mirrored to, and the run resumes.

Correctness at scale comes from the database, not from process-local state,
because any of several app instances may create, mirror, or resolve an ask:

- ``create`` is idempotent per (run_id, tool_call_id): a replayed run returns
  the existing ask rather than asking the same question twice.
- ``resolve`` is a compare-and-swap on ``state``, so when someone answers on
  their phone and their laptop at the same moment, exactly one write wins and
  the other is told it lost.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KIND_APPROVAL = "approval"
KIND_QUESTION = "question"
KIND_PLAN = "plan"

STATE_PENDING = "pending"
STATE_RESOLVED = "resolved"
STATE_CANCELLED = "cancelled"


@dataclass
class Ask:
    """One thing the assistant is waiting on a person for."""

    id: str
    tenant_id: str
    run_id: str
    tool_call_id: str
    kind: str
    title: str = ""
    body: str = ""
    options: List[str] = field(default_factory=list)
    state: str = STATE_PENDING
    resolution: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.state == STATE_PENDING


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return value


def _row_to_ask(row: Any) -> Ask:
    return Ask(
        id=row["id"],
        tenant_id=row["tenant_id"],
        run_id=row["run_id"],
        tool_call_id=row["tool_call_id"],
        kind=row["kind"],
        title=row["title"] or "",
        body=row["body"] or "",
        options=_loads(row["options"], []),
        state=row["state"],
        resolution=row["resolution"],
        data=_loads(row["data"], {}),
    )


class InboxStore:
    """Durable store for pending asks. A None database disables the Inbox."""

    def __init__(self, database: Optional[Any] = None) -> None:
        self._db = database

    @property
    def enabled(self) -> bool:
        return self._db is not None

    async def create(
        self,
        *,
        tenant_id: str,
        run_id: str,
        tool_call_id: str,
        kind: str,
        title: str = "",
        body: str = "",
        options: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Ask]:
        """Record an ask, or return the existing one for this tool call.

        Idempotent by (run_id, tool_call_id): resuming a run that already asked
        must not ask again. The returned ask may therefore already be resolved,
        which is how a resume discovers the answer it was waiting for.
        """
        if not self._db:
            return None
        ask_id = uuid.uuid4().hex
        try:
            row = await self._db.fetchrow(
                """
                INSERT INTO pending_asks
                    (id, tenant_id, run_id, tool_call_id, kind, title, body, options, data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb)
                ON CONFLICT (run_id, tool_call_id) DO UPDATE
                    SET tool_call_id = pending_asks.tool_call_id
                RETURNING id, tenant_id, run_id, tool_call_id, kind, title, body,
                          options, state, resolution, data
                """,
                ask_id,
                tenant_id,
                run_id,
                tool_call_id,
                kind,
                title,
                body,
                json.dumps(options or []),
                json.dumps(data or {}, default=str),
            )
        except Exception as e:
            logger.error(f"Could not create ask for run {run_id}: {e}", exc_info=True)
            return None
        if row is None:
            return None
        ask = _row_to_ask(row)
        if ask.id == ask_id:
            logger.info(f"[Inbox] Asked {tenant_id}: {title or kind} (run={run_id})")
        return ask

    async def resolve(
        self,
        ask_id: str,
        resolution: str,
        *,
        resolved_by: str = "",
    ) -> bool:
        """Answer an ask. Returns False if someone already answered it.

        The WHERE clause on state is what makes this safe across instances:
        the database picks the winner, so two devices answering at once cannot
        both resume the run.
        """
        if not self._db:
            return False
        try:
            row = await self._db.fetchrow(
                """
                UPDATE pending_asks
                   SET state = $2, resolution = $3, resolved_by = $4, resolved_at = NOW()
                 WHERE id = $1 AND state = $5
                RETURNING id
                """,
                ask_id,
                STATE_RESOLVED,
                resolution,
                resolved_by,
                STATE_PENDING,
            )
        except Exception as e:
            logger.error(f"Could not resolve ask {ask_id}: {e}", exc_info=True)
            return False
        if row is None:
            logger.info(f"[Inbox] Ask {ask_id} was already resolved; ignoring")
            return False
        return True

    async def get(self, ask_id: str) -> Optional[Ask]:
        if not self._db:
            return None
        try:
            row = await self._db.fetchrow(
                """
                SELECT id, tenant_id, run_id, tool_call_id, kind, title, body,
                       options, state, resolution, data
                FROM pending_asks WHERE id = $1
                """,
                ask_id,
            )
        except Exception as e:
            logger.warning(f"Could not read ask {ask_id}: {e}")
            return None
        return _row_to_ask(row) if row else None

    async def pending(self, tenant_id: str, limit: int = 50) -> List[Ask]:
        """Everything still waiting on this person, newest first."""
        if not self._db:
            return []
        try:
            rows = await self._db.fetch(
                """
                SELECT id, tenant_id, run_id, tool_call_id, kind, title, body,
                       options, state, resolution, data
                FROM pending_asks
                WHERE tenant_id = $1 AND state = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                tenant_id,
                STATE_PENDING,
                limit,
            )
        except Exception as e:
            logger.warning(f"Could not list pending asks for {tenant_id}: {e}")
            return []
        return [_row_to_ask(r) for r in rows]

    async def for_run(self, run_id: str) -> List[Ask]:
        """Every ask belonging to a run, in creation order.

        A resume reads these to learn which of its outstanding tool calls have
        answers waiting.
        """
        if not self._db:
            return []
        try:
            rows = await self._db.fetch(
                """
                SELECT id, tenant_id, run_id, tool_call_id, kind, title, body,
                       options, state, resolution, data
                FROM pending_asks WHERE run_id = $1 ORDER BY created_at
                """,
                run_id,
            )
        except Exception as e:
            logger.warning(f"Could not list asks for run {run_id}: {e}")
            return []
        return [_row_to_ask(r) for r in rows]

    async def cancel_run(self, run_id: str, reason: str = "run cancelled") -> int:
        """Close a run's open asks. An ask nobody can act on is noise."""
        if not self._db:
            return 0
        try:
            rows = await self._db.fetch(
                """
                UPDATE pending_asks
                   SET state = $2, resolution = $3, resolved_at = NOW()
                 WHERE run_id = $1 AND state = $4
                RETURNING id
                """,
                run_id,
                STATE_CANCELLED,
                reason,
                STATE_PENDING,
            )
            return len(rows)
        except Exception as e:
            logger.warning(f"Could not cancel asks for run {run_id}: {e}")
            return 0
