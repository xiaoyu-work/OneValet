"""SensingAgent — common base for background perception agents.

Subclasses override ``analyze(db, user_id, local_date, tz_name)`` and return
a ``SensingResult`` with user_state fields + optional memory proposals.

Design notes:
  * All sensing agents run with ``expose_as_tool=False``.  They are invoked
    by CronService, never by the LLM.
  * They never send LLM messages themselves — cost stays $0/run.  When an
    agent needs judgment (e.g., interpreting a mood note), it should *emit*
    an item for the weekly reflector to consider.
  * Analysis windows default to "yesterday" in the user's local time; this
    aligns with how humans think about days ("I slept badly last night")
    and sidesteps the partial-day ambiguity of live queries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from koa.standard_agent import StandardAgent

logger = logging.getLogger(__name__)


@dataclass
class SensingResult:
    """Return value from subclasses' ``analyze()``.

    ``user_state_fields`` is upserted verbatim; keys must correspond to
    columns in the ``user_state`` table (see db/021_sensing_foundation.sql).
    ``proposals`` piggy-backs on the existing true_memory_proposals pipe.
    ``flags`` ends up in user_state.flags for downstream fan-out.
    """
    user_state_fields: Dict[str, Any] = field(default_factory=dict)
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    notes: str = ""


class SensingAgent(StandardAgent):
    """Base class for background sensing agents.

    Subclasses MUST override ``analyze()``; they SHOULD NOT override
    ``on_running()`` unless they need custom status reporting.
    """

    tools = ()
    max_turns = 1

    # Subclasses override: which table this agent primarily reads.
    SOURCE_TABLE: str = ""

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _get_db(self):
        return (self.context_hints or {}).get("db")

    def _get_user_id(self) -> Optional[str]:
        uid = (self.context_hints or {}).get("user_id")
        if uid:
            return uid
        return (self.metadata or {}).get("user_id")

    def _yesterday_local(self) -> Tuple[date, str]:
        now, tz_name = self._user_now()
        return (now.date() - timedelta(days=1)), tz_name

    # ------------------------------------------------------------------
    # Subclass entrypoint
    # ------------------------------------------------------------------

    async def analyze(
        self,
        db: Any,
        user_id: str,
        local_date: date,
        tz_name: str,
    ) -> SensingResult:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    async def on_running(self, msg):
        db = self._get_db()
        user_id = self._get_user_id()
        if not db or not user_id:
            logger.warning(
                "%s: missing db or user_id (db=%s user_id=%s); skipping run",
                type(self).__name__, bool(db), bool(user_id),
            )
            return self.make_result(status="skipped", reason="no_context")

        local_date, tz_name = self._yesterday_local()
        try:
            result = await self.analyze(db, user_id, local_date, tz_name)
        except Exception as e:
            logger.exception("%s analyze failed: %s", type(self).__name__, e)
            return self.make_result(status="error", reason=str(e))

        if result.user_state_fields or result.flags:
            fields = dict(result.user_state_fields)
            if result.flags:
                fields["flags"] = result.flags
            fields["timezone"] = tz_name
            await self._upsert_user_state(db, user_id, local_date, fields)

        if result.proposals:
            if self.metadata is None:
                self.metadata = {}
            existing = self.metadata.get("true_memory_proposals", [])
            self.metadata["true_memory_proposals"] = existing + result.proposals

        return self.make_result(
            status="ok",
            summary=result.notes or f"{type(self).__name__} analyzed {local_date}",
            user_state_fields=result.user_state_fields,
            flags=result.flags,
        )

    async def _upsert_user_state(self, db, user_id: str, local_date: date, fields: Dict[str, Any]):
        allowed = {
            "timezone", "sleep_minutes", "sleep_score", "hrv_ms", "resting_hr",
            "steps", "activity_minutes", "stress_score", "mood",
            "primary_location", "focus_mode", "flags", "source_data",
        }
        clean = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not clean:
            return
        cols = ["user_id", "local_date"] + list(clean.keys())
        placeholders = [f"${i+1}" for i in range(len(cols))]
        values = [user_id, local_date] + list(clean.values())
        set_clauses = [f"{k} = EXCLUDED.{k}" for k in clean.keys()]
        sql = (
            f"INSERT INTO user_state ({', '.join(cols)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"ON CONFLICT (user_id, local_date) DO UPDATE SET "
            f"{', '.join(set_clauses)}, updated_at = NOW()"
        )
        try:
            await db.execute(sql, *values)
        except Exception as e:
            logger.error("user_state upsert failed: %s", e)


def make_proposal(
    namespace: str,
    fact_key: str,
    value: Any,
    summary: str,
    *,
    how_to_apply: str = "",
    confidence: float = 0.6,
    why: str = "",
) -> Dict[str, Any]:
    """Helper that shapes a proposal consistent with habit_discovery's format."""
    return {
        "operation": "upsert",
        "namespace": namespace,
        "fact_key": fact_key,
        "value": value,
        "summary": summary,
        "confidence": confidence,
        "source_type": "system_inferred",
        "how_to_apply": how_to_apply,
        "why": why or "Inferred from user sensor data.",
    }
