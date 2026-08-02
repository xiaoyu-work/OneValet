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

    @property
    def is_open(self) -> bool:
        return self.state == STATE_PENDING

    @property
    def awaits_execution(self) -> bool:
        """Approved by the user, but the action has not happened yet."""
        return self.state == STATE_RESOLVED and self.executed_at is None


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
    """
    try:
        rendered = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = str(args)
    return f"{tool_name}\x00{rendered}"


_COLUMNS = """id, tenant_id, run_id, tool_call_id, kind, title, body,
                       options, state, resolution, data, action_key, executed_at"""

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
                     options, data, action_key)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10)
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
                WHERE tenant_id = $1 AND state = $2 AND executed_at IS NULL
                ORDER BY run_id
                LIMIT $3
                """,
                tenant_id,
                STATE_RESOLVED,
                limit,
            )
        except Exception as e:
            logger.warning(f"Could not list runs awaiting execution for {tenant_id}: {e}")
            return []
        return [r["run_id"] for r in rows]

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
