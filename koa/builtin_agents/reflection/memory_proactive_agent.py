"""MemoryProactiveAgent — consumes weekly reflection + user_state to decide
whether a *memory-driven* proactive notification is warranted.

Distinct from the existing ProactiveCheckAgent (which looks at calendar /
tasks / subscriptions for time-sensitive alerts).  This agent's trigger
surface is the user's *patterns* — anniversaries, routine breaks, milestones.

Runs daily but sends at most 1 notification per day; if multiple candidates
exist it picks the highest-importance one.

Candidate sources:
  * episodes with local_date == today - 365 (anniversary)
  * episodes with local_date == today - 7/30 (weekly/monthly retrospectives)
  * latest weekly_reflection.highlight if generated today (first Monday push)
  * user_state flags of last 3 days (e.g., "low_sleep" 3/3 → caring nudge)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from koa import valet
from koa.standard_agent import StandardAgent

logger = logging.getLogger(__name__)


@valet(domain="reflection", expose_as_tool=False)
class MemoryProactiveAgent(StandardAgent):
    """Daily check: should we surface a memory-driven nudge to the user?"""

    tools = ()
    max_turns = 1

    async def on_running(self, msg):
        db = (self.context_hints or {}).get("db")
        user_id = (self.context_hints or {}).get("user_id") or (self.metadata or {}).get("user_id")
        if not db or not user_id:
            return self.make_result(status="skipped", reason="no_context")

        now, _tz = self._user_now()
        today = now.date()

        candidate = await _select_candidate(db, user_id, today)
        if candidate is None:
            return self.make_result(status="nothing_to_report")

        # Hand off to orchestrator/push delivery via result metadata.
        # The orchestrator has logic to route `notification` payloads into
        # the user's preferred channel (push / SMS / in-app).
        return self.make_result(
            status="ok",
            notification={
                "kind": candidate["kind"],
                "title": candidate["title"],
                "body": candidate["body"],
                "payload": candidate.get("payload", {}),
            },
            summary=f"Memory-driven nudge ({candidate['kind']}): {candidate['title']}",
        )


async def _select_candidate(db, user_id: str, today: date) -> Optional[Dict[str, Any]]:
    """Rank candidates by importance, return the best one.  Returns None if
    nothing meets threshold.

    Rules:
      * 1y anniversary of importance>=4 episode → kind=anniversary
      * routine break in last 7 days (episode kind='routine_break') → kind=checkin
      * 3 of last 3 days flagged low_sleep → kind=wellbeing
    """
    candidates: List[Tuple[int, Dict[str, Any]]] = []

    # 1y anniversary of a meaningful episode
    try:
        row = await db.fetchrow(
            """SELECT title, summary FROM episodes
               WHERE user_id = $1 AND status = 'active'
                 AND local_date = $2 AND importance >= 4
               ORDER BY importance DESC LIMIT 1""",
            user_id, today - timedelta(days=365),
        )
        if row:
            candidates.append((90, {
                "kind": "anniversary",
                "title": "One year ago today",
                "body": f"{row['title']}. {row['summary'][:120]}",
                "payload": {"type": "anniversary"},
            }))
    except Exception as e:
        logger.debug("anniversary check failed: %s", e)

    # Recent routine break
    try:
        row = await db.fetchrow(
            """SELECT title, summary FROM episodes
               WHERE user_id = $1 AND status = 'active'
                 AND kind = 'routine_break'
                 AND local_date >= $2
               ORDER BY local_date DESC LIMIT 1""",
            user_id, today - timedelta(days=6),
        )
        if row:
            candidates.append((60, {
                "kind": "checkin",
                "title": "Everything ok?",
                "body": f"Noticed a change recently: {row['title']}.",
                "payload": {"type": "routine_break"},
            }))
    except Exception as e:
        logger.debug("routine_break check failed: %s", e)

    # Low-sleep streak (wellbeing)
    try:
        rows = await db.fetch(
            """SELECT flags FROM user_state
               WHERE user_id = $1 AND local_date >= $2 AND local_date <= $3
               ORDER BY local_date DESC""",
            user_id, today - timedelta(days=2), today,
        )
        low_sleep_days = sum(
            1 for r in rows
            if r.get("flags") and "low_sleep" in (r["flags"] or [])
        )
        if low_sleep_days >= 3:
            candidates.append((70, {
                "kind": "wellbeing",
                "title": "Noticed you've been running light on sleep",
                "body": "Want me to keep today low-key and push non-urgent reminders to tomorrow?",
                "payload": {"type": "low_sleep_streak", "days": low_sleep_days},
            }))
    except Exception as e:
        logger.debug("low_sleep check failed: %s", e)

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]
