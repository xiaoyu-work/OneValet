"""Durable storage for in-flight ReAct transcripts.

A run's transcript is the record of what it has already done: the user's
message, every tool call, and every result. Keeping it lets a run outlive the
process that started it -- whether it stopped because the server restarted or
because the agent needs a human decision it cannot get right now.

The transcript is stored verbatim, in the shape the model is sent, so
resuming replays it rather than reconstructing intent from a summary.

All methods degrade to no-ops when no database is configured, so an
in-memory deployment behaves exactly as before.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: The run is executing right now.
STATUS_RUNNING = "running"
#: The run stopped mid-flight and can be continued -- it is waiting on a human,
#: or the process died before it finished.
STATUS_SUSPENDED = "suspended"
#: Terminal. Kept briefly for audit, then pruned.
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


@dataclass
class RunTranscript:
    run_id: str
    tenant_id: str
    status: str
    messages: List[Dict[str, Any]]
    user_message: str
    metadata: Dict[str, Any]
    turn: int


def _loads(value: Any, default: Any) -> Any:
    """asyncpg returns JSONB as str on some driver versions and dict on others."""
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return value


class TranscriptStore:
    """Reads and writes run transcripts. A None database disables persistence."""

    def __init__(self, database: Optional[Any] = None) -> None:
        self._db = database

    @property
    def enabled(self) -> bool:
        return self._db is not None

    async def save(
        self,
        run_id: str,
        tenant_id: str,
        messages: List[Dict[str, Any]],
        *,
        user_message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        turn: int = 0,
        status: str = STATUS_RUNNING,
    ) -> None:
        """Write the run's current transcript, replacing any previous version.

        Called after each tool round, so the stored copy always reflects work
        that actually completed. A failure here must not break the request --
        the run is still valid in memory, it just will not survive a restart.
        """
        if not self._db:
            return
        try:
            await self._db.execute(
                """
                INSERT INTO run_transcripts
                    (run_id, tenant_id, status, messages, user_message, metadata, turn)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, $7)
                ON CONFLICT (run_id) DO UPDATE SET
                    status       = EXCLUDED.status,
                    messages     = EXCLUDED.messages,
                    user_message = EXCLUDED.user_message,
                    metadata     = EXCLUDED.metadata,
                    turn         = EXCLUDED.turn,
                    updated_at   = NOW()
                """,
                run_id,
                tenant_id,
                status,
                json.dumps(messages, default=str),
                user_message,
                json.dumps(metadata or {}, default=str),
                turn,
            )
        except Exception as e:
            logger.warning(f"Could not persist transcript for run {run_id}: {e}")

    async def get(self, run_id: str) -> Optional[RunTranscript]:
        if not self._db:
            return None
        try:
            row = await self._db.fetchrow(
                """
                SELECT run_id, tenant_id, status, messages, user_message, metadata, turn
                FROM run_transcripts WHERE run_id = $1
                """,
                run_id,
            )
        except Exception as e:
            logger.warning(f"Could not read transcript {run_id}: {e}")
            return None
        if row is None:
            return None
        return RunTranscript(
            run_id=row["run_id"],
            tenant_id=row["tenant_id"],
            status=row["status"],
            messages=_loads(row["messages"], []),
            user_message=row["user_message"] or "",
            metadata=_loads(row["metadata"], {}),
            turn=row["turn"] or 0,
        )

    async def claim(self, run_id: str, stale_after_seconds: int = 900) -> bool:
        """Take exclusive ownership of continuing a run.

        Two answers to the same run can arrive within a second of each other,
        and an operator can POST a resume for a run that is already going. Both
        would replay the same transcript concurrently: two loops making the
        same calls, and whichever finished last overwriting the other's record
        of what happened.

        The state change is the claim, so the database picks the winner. A run
        already marked running is not simply refused, because a process that
        died mid-run leaves that mark behind forever; one that has gone quiet
        for longer than the lease is treated as abandoned and may be taken
        over. A live run keeps its lease fresh by saving each turn.
        """
        if not self._db:
            return False
        try:
            row = await self._db.fetchrow(
                """
                UPDATE run_transcripts
                   SET status = $2, updated_at = NOW()
                 WHERE run_id = $1
                   AND (status = $3
                        OR (status = $2 AND updated_at < NOW() - ($4 || ' seconds')::interval))
                RETURNING run_id
                """,
                run_id,
                STATUS_RUNNING,
                STATUS_SUSPENDED,
                str(stale_after_seconds),
            )
        except Exception as e:
            logger.warning(f"Could not claim run {run_id}: {e}")
            return False
        return row is not None

    async def mark(self, run_id: str, status: str) -> None:
        """Move a run to a terminal or suspended state."""
        if not self._db:
            return
        try:
            await self._db.execute(
                "UPDATE run_transcripts SET status = $2, updated_at = NOW() WHERE run_id = $1",
                run_id,
                status,
            )
        except Exception as e:
            logger.warning(f"Could not mark run {run_id} as {status}: {e}")

    async def prune(self, older_than_hours: int = 48) -> int:
        """Drop finished transcripts. Suspended runs are never pruned here --
        they are still waiting on someone."""
        if not self._db:
            return 0
        try:
            result = await self._db.execute(
                """
                DELETE FROM run_transcripts
                WHERE status IN ($1, $2)
                  AND updated_at < NOW() - ($3 || ' hours')::interval
                """,
                STATUS_COMPLETED,
                STATUS_FAILED,
                str(older_than_hours),
            )
            return int(str(result).rsplit(" ", 1)[-1]) if result else 0
        except Exception as e:
            logger.warning(f"Could not prune transcripts: {e}")
            return 0

    async def list_resumable(
        self,
        tenant_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Runs for this tenant that stopped before reaching a terminal state."""
        if not self._db:
            return []
        try:
            rows = await self._db.fetch(
                """
                SELECT run_id, status, user_message, turn, updated_at
                FROM run_transcripts
                WHERE tenant_id = $1 AND status IN ($2, $3)
                ORDER BY updated_at DESC
                LIMIT $4
                """,
                tenant_id,
                STATUS_RUNNING,
                STATUS_SUSPENDED,
                limit,
            )
        except Exception as e:
            logger.warning(f"Could not list resumable runs for {tenant_id}: {e}")
            return []
        return [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "user_message": r["user_message"],
                "turn": r["turn"],
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            }
            for r in rows
        ]

    @staticmethod
    def unanswered_tool_calls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tool calls from the final assistant turn that have no result yet.

        This is what a resume has to finish. Answered calls are skipped, so
        replaying is idempotent: a run interrupted after two of three tools ran
        does not re-run those two.
        """
        answered = {
            m.get("tool_call_id")
            for m in messages
            if isinstance(m, dict) and m.get("role") == "tool"
        }
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                return []
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                return [
                    tc
                    for tc in msg["tool_calls"]
                    if isinstance(tc, dict) and tc.get("id") not in answered
                ]
        return []
