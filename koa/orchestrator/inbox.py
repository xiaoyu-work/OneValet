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

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KIND_APPROVAL = "approval"

STATE_PENDING = "pending"
STATE_RESOLVED = "resolved"
STATE_EXPIRED = "expired"


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

    #: Identity of the action this is about -- see ``action_key``.
    action_key: str = ""

    #: When the approved action was carried out. None means it still owes one.
    executed_at: Optional[Any] = None
    expires_at: Optional[Any] = None

    @property
    def is_open(self) -> bool:
        if self.state != STATE_PENDING:
            return False
        if self.expires_at is None:
            return True
        expires_at = self.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at > datetime.now(timezone.utc)

    @property
    def awaits_execution(self) -> bool:
        """Approved by the user, but the action has not happened yet."""
        return (
            self.kind == KIND_APPROVAL
            and self.state == STATE_RESOLVED
            and self.executed_at is None
        )


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return value


def action_key(tool_name: Any, args: Any) -> str:
    """Identity of an action, stable across the runs that ask about it.

    An approval outlives the run that requested it, so the run that acts on
    it is a different process rebuilding the same arguments from scratch.
    They are matched by value, which means both sides have to be rendered the
    same way -- including the trip through the database, where a value that
    is not JSON has already been flattened to a string by ``default=str``.
    Rendering the live side any other way would leave a run unable to
    recognise its own approval, and it would ask again.

    The stored key is a versioned SHA-256 hex digest rather than a delimited
    string. PostgreSQL TEXT rejects NUL bytes, and a separator-based key also
    has to prove that neither half can forge the boundary; canonical JSON plus
    a digest has neither problem.
    """
    try:
        rendered = json.dumps(
            [str(tool_name), args],
            sort_keys=True,
            default=str,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        rendered = repr([str(tool_name), args])
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return f"v1:{digest}"


_COLUMNS = """id, tenant_id, run_id, tool_call_id, kind, title, body,
                       options, state, resolution, data, action_key, executed_at,
                       expires_at"""

#: Answers that mean "go ahead". Anything else is a refusal, so an answer we
#: do not understand stops the action rather than performing it.
_AFFIRMATIVE = {"approve", "approved", "yes", "y", "ok", "okay", "confirm", "allow", "accept"}


def is_approval(resolution: Optional[str]) -> bool:
    """Whether a recorded answer authorises the action."""
    return (resolution or "").strip().lower() in _AFFIRMATIVE


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
        action_key=_key_of(row),
        executed_at=_get(row, "executed_at"),
        expires_at=_get(row, "expires_at"),
    )


def _get(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        return None


def _key_of(row: Any) -> str:
    return _get(row, "action_key") or ""


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
        action_key: str,
        expires_in_seconds: int = 604800,
        title: str = "",
        body: str = "",
        options: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Ask]:
        """Record an ask, or return the existing one for this action.

        Idempotent by (run_id, action_key) -- the identity of the action
        itself, not of the tool call that happened to raise it. A replayed run
        mints new tool call ids, so keying on those would let the same
        question be asked again; keying on the action means a replay finds the
        answer instead. The returned ask may therefore already be resolved,
        which is how a resume discovers what the user decided.
        """
        if not self._db:
            return None
        ask_id = uuid.uuid4().hex
        try:
            row = await self._db.fetchrow(
                f"""
                INSERT INTO pending_asks
                    (id, tenant_id, run_id, tool_call_id, kind, title, body,
                     options, data, action_key, expires_at)
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10,
                    NOW() + ($11 || ' seconds')::interval
                )
                ON CONFLICT (run_id, action_key) DO UPDATE
                    SET action_key = pending_asks.action_key
                RETURNING {_COLUMNS}
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
                action_key,
                str(int(expires_in_seconds)),
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

    async def claim_execution(self, ask_id: str) -> bool:
        """Take ownership of carrying out an approved action.

        An approval authorises one action, once. Two resumes of the same run
        can race here, so the stamp is the claim: whoever sets executed_at
        performs the action and everyone else is told they lost.
        """
        if not self._db:
            return False
        try:
            row = await self._db.fetchrow(
                """
                UPDATE pending_asks
                   SET executed_at = NOW()
                 WHERE id = $1 AND state = $2 AND executed_at IS NULL
                RETURNING id
                """,
                ask_id,
                STATE_RESOLVED,
            )
        except Exception as e:
            logger.error(f"Could not claim execution of ask {ask_id}: {e}", exc_info=True)
            return False
        return row is not None

    async def runs_awaiting_execution(self, tenant_id: str, limit: int = 20) -> List[str]:
        """Runs holding a decision the user made that was never acted on.

        A run reaches this state when the model, on resuming, did not go back
        to the agent that owns the approved action. Nothing else will notice:
        both ways of waking a run start from answering a question that is
        still open, and this question has been answered. Without a way to list
        them, the user's decision sits in the database forever.
        """
        if not self._db:
            return []
        try:
            rows = await self._db.fetch(
                """
                SELECT DISTINCT run_id FROM pending_asks
                WHERE tenant_id = $1
                  AND kind = $2
                  AND state = $3
                  AND executed_at IS NULL
                ORDER BY run_id
                LIMIT $4
                """,
                tenant_id,
                KIND_APPROVAL,
                STATE_RESOLVED,
                limit,
            )
        except Exception as e:
            logger.warning(f"Could not list runs awaiting execution for {tenant_id}: {e}")
            return []
        return [r["run_id"] for r in rows]

    async def recoverable_runs(
        self,
        stale_after_seconds: int,
        max_attempts: int,
        limit: int = 100,
    ) -> List[str]:
        """Suspended runs whose answered decisions are ready to carry out.

        Internal maintenance query, intentionally across tenants. The run
        itself carries its tenant and every continuation is fenced by the
        transcript claim, so several app instances may discover the same row
        without executing it twice.
        """
        if not self._db:
            return []
        try:
            rows = await self._db.fetch(
                """
                SELECT pa.run_id
                FROM pending_asks AS pa
                JOIN run_transcripts AS rt ON rt.run_id = pa.run_id
                WHERE pa.state = $1
                  AND pa.kind = $2
                  AND pa.executed_at IS NULL
                  AND (
                      rt.status = $3
                      OR (
                          rt.status = $4
                          AND rt.updated_at
                              < NOW() - ($5 || ' seconds')::interval
                      )
                  )
                  AND rt.recovery_attempts < $7
                  AND (
                      rt.next_recovery_at IS NULL
                      OR rt.next_recovery_at <= NOW()
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM pending_asks AS open_ask
                      WHERE open_ask.run_id = pa.run_id
                        AND open_ask.state = $6
                        AND open_ask.expires_at > NOW()
                  )
                GROUP BY pa.run_id, rt.updated_at
                ORDER BY rt.updated_at
                LIMIT $8
                """,
                STATE_RESOLVED,
                KIND_APPROVAL,
                "suspended",
                "running",
                str(int(stale_after_seconds)),
                STATE_PENDING,
                max_attempts,
                limit,
            )
        except Exception as e:
            logger.warning(f"Could not list recoverable Inbox runs: {e}")
            return []
        return [r["run_id"] for r in rows]

    async def expire(self, batch_size: int = 5000) -> int:
        """Expire unanswered asks and close runs that have nothing else owed."""
        if not self._db:
            return 0
        try:
            row = await self._db.fetchrow(
                """
                WITH candidates AS (
                    SELECT id FROM pending_asks
                    WHERE state = $2 AND expires_at <= NOW()
                    ORDER BY expires_at
                    LIMIT $6
                    FOR UPDATE SKIP LOCKED
                ),
                expired AS (
                    UPDATE pending_asks
                       SET state = $1,
                           resolution = $1,
                           resolved_at = NOW()
                      FROM candidates
                     WHERE pending_asks.id = candidates.id
                    RETURNING run_id
                ),
                closed_runs AS (
                    UPDATE run_transcripts AS rt
                       SET status = $3, updated_at = NOW()
                     WHERE rt.status = $4
                       AND rt.run_id IN (SELECT run_id FROM expired)
                       AND NOT EXISTS (
                           SELECT 1 FROM pending_asks AS open_ask
                           WHERE open_ask.run_id = rt.run_id
                             AND open_ask.state = $2
                             AND open_ask.expires_at > NOW()
                       )
                       AND NOT EXISTS (
                           SELECT 1 FROM pending_asks AS unexecuted
                           WHERE unexecuted.run_id = rt.run_id
                             AND unexecuted.state = $5
                             AND unexecuted.executed_at IS NULL
                       )
                    RETURNING rt.run_id
                )
                SELECT COUNT(*) AS expired_count FROM expired
                """,
                STATE_EXPIRED,
                STATE_PENDING,
                "failed",
                "suspended",
                STATE_RESOLVED,
                batch_size,
            )
            return int(row["expired_count"]) if row else 0
        except Exception as e:
            logger.warning(f"Could not expire Inbox asks: {e}")
            return 0

    async def prune(self, older_than_days: int = 30, batch_size: int = 1000) -> int:
        """Delete terminal asks in bounded batches.

        Open asks and answered-but-unexecuted decisions are never deleted:
        either still represents work owed to a person. Rows whose decisions
        were carried out are audit history, retained for a month and then
        removed so one approval-gated call does not mean one permanent row.
        """
        if not self._db:
            return 0
        try:
            result = await self._db.execute(
                """
                WITH doomed AS (
                    SELECT id FROM pending_asks
                    WHERE (
                              (state = $1 AND executed_at IS NOT NULL)
                              OR state = $2
                          )
                      AND resolved_at < NOW() - ($3 || ' days')::interval
                    ORDER BY resolved_at
                    LIMIT $4
                          FOR UPDATE SKIP LOCKED
                )
                DELETE FROM pending_asks AS pa
                USING doomed
                WHERE pa.id = doomed.id
                """,
                STATE_RESOLVED,
                STATE_EXPIRED,
                str(int(older_than_days)),
                batch_size,
            )
            return int(str(result).rsplit(" ", 1)[-1]) if result else 0
        except Exception as e:
            logger.warning(f"Could not prune Inbox asks: {e}")
            return 0

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
                WITH resolved AS (
                    UPDATE pending_asks
                       SET state = $2,
                           resolution = $3,
                           resolved_by = $4,
                           resolved_at = NOW()
                     WHERE id = $1 AND state = $5 AND expires_at > NOW()
                    RETURNING id, run_id
                ),
                reset_budget AS (
                    UPDATE run_transcripts AS rt
                       SET recovery_attempts = 0,
                           next_recovery_at = NULL
                      FROM resolved
                     WHERE rt.run_id = resolved.run_id
                )
                SELECT id FROM resolved
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
                f"""
                SELECT {_COLUMNS}
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
                f"""
                SELECT {_COLUMNS}
                FROM pending_asks
                WHERE tenant_id = $1 AND state = $2 AND expires_at > NOW()
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
                f"""
                SELECT {_COLUMNS}
                FROM pending_asks WHERE run_id = $1 ORDER BY created_at
                """,
                run_id,
            )
        except Exception as e:
            logger.warning(f"Could not list asks for run {run_id}: {e}")
            return []
        return [_row_to_ask(r) for r in rows]
