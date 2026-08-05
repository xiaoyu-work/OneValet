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
import uuid
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


class RunLeaseLost(RuntimeError):
    """The transcript's fencing token no longer belongs to this process."""


@dataclass
class RunTranscript:
    run_id: str
    tenant_id: str
    status: str
    messages: List[Dict[str, Any]]
    user_message: str
    metadata: Dict[str, Any]
    turn: int
    claim_token: Optional[str] = None
    recovery_attempts: int = 0
    next_recovery_at: Optional[Any] = None


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


def _changed(result: Any) -> bool:
    """Whether an asyncpg execute result changed at least one row."""
    if not result:
        return False
    try:
        return int(str(result).rsplit(" ", 1)[-1]) > 0
    except (TypeError, ValueError):
        return False


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
        claim_token: Optional[str] = None,
    ) -> bool:
        """Write the run's current transcript, replacing any previous version.

        Called before a tool round to make its unanswered calls recoverable,
        and after results arrive to record what actually completed. A failure
        on a claimed run fences execution: continuing without a transcript
        would let another process take over and repeat side effects.
        """
        if not self._db:
            return False
        try:
            if claim_token:
                result = await self._db.execute(
                    """
                    UPDATE run_transcripts SET
                        status       = $3,
                        messages     = $4::jsonb,
                        user_message = $5,
                        metadata     = $6::jsonb,
                        turn         = $7,
                        updated_at   = NOW()
                    WHERE run_id = $1 AND tenant_id = $2 AND claim_token = $8
                    """,
                    run_id,
                    tenant_id,
                    status,
                    json.dumps(messages, default=str),
                    user_message,
                    json.dumps(metadata or {}, default=str),
                    turn,
                    claim_token,
                )
            else:
                result = await self._db.execute(
                    """
                    INSERT INTO run_transcripts
                        (run_id, tenant_id, status, messages, user_message, metadata,
                         turn, claim_token)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, $7, NULL)
                    ON CONFLICT (run_id) DO UPDATE SET
                        status       = EXCLUDED.status,
                        messages     = EXCLUDED.messages,
                        user_message = EXCLUDED.user_message,
                        metadata     = EXCLUDED.metadata,
                        turn         = EXCLUDED.turn,
                        updated_at   = NOW()
                    WHERE run_transcripts.claim_token IS NULL
                    """,
                    run_id,
                    tenant_id,
                    status,
                    json.dumps(messages, default=str),
                    user_message,
                    json.dumps(metadata or {}, default=str),
                    turn,
                )
            return _changed(result)
        except Exception as e:
            logger.warning(f"Could not persist transcript for run {run_id}: {e}")
            return False

    async def get(self, run_id: str) -> Optional[RunTranscript]:
        if not self._db:
            return None
        try:
            row = await self._db.fetchrow(
                """
                SELECT run_id, tenant_id, status, messages, user_message, metadata,
                       turn, claim_token, recovery_attempts, next_recovery_at
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
            claim_token=row["claim_token"],
            recovery_attempts=row["recovery_attempts"] or 0,
            next_recovery_at=row["next_recovery_at"],
        )

    async def claim(self, run_id: str, stale_after_seconds: int) -> Optional[str]:
        """Take exclusive ownership of continuing a run.

        Two answers to the same run can arrive within a second of each other,
        and an operator can ask to resume a run that is already going. Both
        would replay the same transcript concurrently: two loops making the
        same calls, and whichever finished last overwriting the other's record
        of what happened.

        The state change is the claim, so the database picks the winner. A run
        already marked running is not simply refused, because a process that
        died mid-run leaves that mark behind forever; one that has gone quiet
        for longer than the lease is treated as abandoned and may be taken
        over. A live run keeps its lease fresh by saving each turn.

        The caller sets the lease, because only it knows how long a turn can
        legitimately take. Erring long costs a slower recovery from a crash;
        erring short means stealing a run that is still working, and running
        its tools a second time.
        """
        if not self._db:
            return None
        token = uuid.uuid4().hex
        try:
            row = await self._db.fetchrow(
                """
                UPDATE run_transcripts
                   SET status = $2, claim_token = $5, updated_at = NOW()
                 WHERE run_id = $1
                   AND (status = $3
                        OR (status = $2 AND updated_at < NOW() - ($4 || ' seconds')::interval))
                RETURNING run_id
                """,
                run_id,
                STATUS_RUNNING,
                STATUS_SUSPENDED,
                str(int(stale_after_seconds)),
                token,
            )
        except Exception as e:
            logger.warning(f"Could not claim run {run_id}: {e}")
            return None
        return token if row is not None else None

    async def reserve_recovery(
        self,
        run_id: str,
        stale_after_seconds: int,
        max_attempts: int,
        base_delay_seconds: int,
    ) -> bool:
        """Reserve one automatic recovery attempt with durable backoff.

        The update is the multi-instance arbiter. A process-local counter would
        reset on deploy and every app instance would get its own budget.
        """
        if not self._db:
            return False
        try:
            row = await self._db.fetchrow(
                """
                UPDATE run_transcripts
                   SET recovery_attempts = recovery_attempts + 1,
                       next_recovery_at = NOW() + make_interval(
                           secs => $6::integer * power(2, recovery_attempts)::integer
                       )
                 WHERE run_id = $1
                   AND recovery_attempts < $5
                   AND (next_recovery_at IS NULL OR next_recovery_at <= NOW())
                   AND (
                       status = $3
                       OR (
                           status = $2
                           AND updated_at
                               < NOW() - ($4 || ' seconds')::interval
                       )
                   )
                RETURNING run_id
                """,
                run_id,
                STATUS_RUNNING,
                STATUS_SUSPENDED,
                str(int(stale_after_seconds)),
                max_attempts,
                base_delay_seconds,
            )
        except Exception as e:
            logger.warning(f"Could not reserve recovery of run {run_id}: {e}")
            return False
        return row is not None

    async def release(self, run_id: str, claim_token: str) -> None:
        """Hand a claimed run back, but only if it is still ours to hand back.

        The loop sets the run's final status before the caller has finished
        with it, so a failure in that tail must not rewrite a run that
        completed into one that looks unfinished. Only a row still marked
        running is one nobody has concluded.
        """
        if not self._db:
            return
        try:
            await self._db.execute(
                """
                UPDATE run_transcripts
                   SET status = $2, updated_at = NOW()
                 WHERE run_id = $1 AND status = $3 AND claim_token = $4
                """,
                run_id,
                STATUS_SUSPENDED,
                STATUS_RUNNING,
                claim_token,
            )
        except Exception as e:
            logger.warning(f"Could not release run {run_id}: {e}")

    async def touch(self, run_id: str, claim_token: Optional[str] = None) -> bool:
        """Say the run is still alive, without rewriting its transcript.

        The lease is what stops a second caller taking over a run that is
        still working, and it is only as good as the last time the run said
        anything. A turn that ends without recording new work -- every tool
        call rejected, say -- still needs to count as a sign of life.
        """
        if not self._db:
            return False
        try:
            if claim_token:
                result = await self._db.execute(
                    "UPDATE run_transcripts SET updated_at = NOW() "
                    "WHERE run_id = $1 AND status = $2 AND claim_token = $3",
                    run_id,
                    STATUS_RUNNING,
                    claim_token,
                )
            else:
                result = await self._db.execute(
                    "UPDATE run_transcripts SET updated_at = NOW() "
                    "WHERE run_id = $1 AND status = $2 AND claim_token IS NULL",
                    run_id,
                    STATUS_RUNNING,
                )
            return _changed(result)
        except Exception as e:
            # The concurrency guarantee rests on this: a touch that keeps
            # failing lets a live run look abandoned and be taken over.
            logger.warning(f"Could not touch run {run_id}: {e}")
            return False

    async def mark(
        self,
        run_id: str,
        status: str,
        claim_token: Optional[str] = None,
    ) -> bool:
        """Move a run to a terminal or suspended state."""
        if not self._db:
            return False
        try:
            if claim_token:
                result = await self._db.execute(
                    """
                    UPDATE run_transcripts
                       SET status = $2,
                           updated_at = NOW(),
                           recovery_attempts = CASE WHEN $2 = $4 THEN 0
                                                    ELSE recovery_attempts END,
                           next_recovery_at = CASE WHEN $2 = $4 THEN NULL
                                                   ELSE next_recovery_at END
                     WHERE run_id = $1 AND claim_token = $3
                    """,
                    run_id,
                    status,
                    claim_token,
                    STATUS_COMPLETED,
                )
            else:
                result = await self._db.execute(
                    """
                    UPDATE run_transcripts
                       SET status = $2,
                           updated_at = NOW(),
                           recovery_attempts = CASE WHEN $2 = $3 THEN 0
                                                    ELSE recovery_attempts END,
                           next_recovery_at = CASE WHEN $2 = $3 THEN NULL
                                                   ELSE next_recovery_at END
                     WHERE run_id = $1 AND claim_token IS NULL
                    """,
                    run_id,
                    status,
                    STATUS_COMPLETED,
                )
            return _changed(result)
        except Exception as e:
            logger.warning(f"Could not mark run {run_id} as {status}: {e}")
            return False

    async def fail(self, run_id: str, claim_token: Optional[str] = None) -> bool:
        """Mark an unfinished run failed without rewriting a finished one."""
        if not self._db:
            return False
        try:
            if claim_token:
                result = await self._db.execute(
                    """
                    UPDATE run_transcripts
                       SET status = $2, updated_at = NOW()
                     WHERE run_id = $1
                       AND status = $3
                       AND claim_token = $4
                    """,
                    run_id,
                    STATUS_FAILED,
                    STATUS_RUNNING,
                    claim_token,
                )
            else:
                result = await self._db.execute(
                    """
                    UPDATE run_transcripts
                       SET status = $2, updated_at = NOW()
                     WHERE run_id = $1
                       AND status = $3
                       AND claim_token IS NULL
                    """,
                    run_id,
                    STATUS_FAILED,
                    STATUS_RUNNING,
                )
            return _changed(result)
        except Exception as e:
            logger.warning(f"Could not mark run {run_id} failed: {e}")
            return False

    async def prune(self, older_than_hours: int = 48, batch_size: int = 1000) -> int:
        """Drop terminal and abandoned transcripts in bounded batches.

        A very old running or suspended row is safe to remove only when no
        open or unexecuted ask says the user is still owed something.
        Suspended is also used for user interruption and abandoned resumes,
        neither of which necessarily has anything durable left to recover.
        """
        if not self._db:
            return 0
        try:
            result = await self._db.execute(
                """
                WITH doomed AS (
                    SELECT rt.run_id FROM run_transcripts AS rt
                    WHERE rt.updated_at
                              < NOW() - ($7 || ' hours')::interval
                      AND (
                          rt.status IN ($1, $2)
                          OR (
                              rt.status IN ($3, $4)
                              AND NOT EXISTS (
                                  SELECT 1 FROM pending_asks AS pa
                                  WHERE pa.run_id = rt.run_id
                                    AND (
                                            pa.state = $5
                                        OR (
                                                pa.state = $6
                                            AND pa.executed_at IS NULL
                                        )
                                    )
                              )
                          )
                      )
                    ORDER BY rt.updated_at
                      LIMIT $8
                      FOR UPDATE OF rt SKIP LOCKED
                )
                DELETE FROM run_transcripts AS rt
                USING doomed
                WHERE rt.run_id = doomed.run_id
                """,
                STATUS_COMPLETED,
                STATUS_FAILED,
                STATUS_RUNNING,
                STATUS_SUSPENDED,
                "pending",
                "resolved",
                str(int(older_than_hours)),
                batch_size,
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
                SELECT run_id, status, user_message, turn, updated_at,
                       recovery_attempts, next_recovery_at
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
                "recovery_attempts": r["recovery_attempts"] or 0,
                "next_recovery_at": (
                    r["next_recovery_at"].isoformat() if r["next_recovery_at"] else None
                ),
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
