"""MemoryProactiveAgent — consumes reflection episodes + user_state to
decide whether a *memory-driven* proactive notification is warranted.

Distinct from the existing ProactiveCheckAgent (calendar/tasks/subscriptions).
This agent's trigger surface is the user's *patterns* — anniversaries,
routine breaks, milestones.

Runs daily; sends at most 1 notification per day (picks highest importance).

Candidate sources:
  * Momex episodes (subkind=behavioral_pattern) with local_date == today-365
    → anniversary.
  * Momex episodes (kind_hint=routine_break) from the last 7 days → checkin.
  * user_state flags in the last 3 days with "low_sleep" 3/3 → wellbeing.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from koa import valet
from koa.memory.lifecycle.episode_memory import EpisodeMemory
from koa.standard_agent import StandardAgent

logger = logging.getLogger(__name__)


@valet(domain="reflection", expose_as_tool=False)
class MemoryProactiveAgent(StandardAgent):
    """Daily check: should we surface a memory-driven nudge to the user?"""

    tools = ()
    max_turns = 1

    async def on_running(self, msg):
        hints = self.context_hints or {}
        db = hints.get("db")
        momex = hints.get("momex")
        user_id = hints.get("user_id") or (self.metadata or {}).get("user_id")
        if not user_id:
            return self.make_result(status="skipped", reason="no_context")

        now, _tz = self._user_now()
        today = now.date()

        episode_memory = EpisodeMemory(momex) if momex is not None else None
        candidate = await _select_candidate(db, episode_memory, user_id, today)
        if candidate is None:
            return self.make_result(status="nothing_to_report")

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


async def _select_candidate(
    db, episode_memory: Optional[EpisodeMemory], user_id: str, today: date
) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[int, Dict[str, Any]]] = []

    # 1-year anniversary of a meaningful episode.
    if episode_memory is not None:
        try:
            items = await episode_memory.recall_episodes(
                user_id,
                "one year ago anniversary",
                limit=10,
                subkind="behavioral_pattern",
                oversample=10,
            )
            target_date = (today - timedelta(days=365)).isoformat()
            for it in items:
                meta = it.get("metadata") or {}
                if meta.get("local_date") != target_date:
                    continue
                if int(meta.get("importance") or 0) < 4:
                    continue
                title = meta.get("title") or "One year ago"
                body = (it.get("text") or "")[:160]
                candidates.append(
                    (
                        90,
                        {
                            "kind": "anniversary",
                            "title": "One year ago today",
                            "body": f"{title}. {body}",
                            "payload": {"type": "anniversary"},
                        },
                    )
                )
                break
        except Exception as e:
            logger.debug("anniversary check failed: %s", e)

    # Recent routine break.
    if episode_memory is not None:
        try:
            items = await episode_memory.recall_episodes(
                user_id,
                "routine break recent change",
                limit=10,
                subkind="behavioral_pattern",
                oversample=10,
            )
            window_start = today - timedelta(days=6)
            for it in items:
                meta = it.get("metadata") or {}
                if meta.get("kind_hint") != "routine_break":
                    continue
                ld_raw = meta.get("local_date")
                try:
                    ld = date.fromisoformat(ld_raw) if ld_raw else None
                except Exception:
                    ld = None
                if ld is None or ld < window_start:
                    continue
                title = meta.get("title") or "Recent change"
                candidates.append(
                    (
                        60,
                        {
                            "kind": "checkin",
                            "title": "Everything ok?",
                            "body": f"Noticed a change recently: {title}.",
                            "payload": {"type": "routine_break"},
                        },
                    )
                )
                break
        except Exception as e:
            logger.debug("routine_break check failed: %s", e)

    # Low-sleep streak (still served from user_state — structured scalars).
    if db is not None:
        try:
            rows = await db.fetch(
                """SELECT flags FROM user_state
                   WHERE user_id = $1 AND local_date >= $2 AND local_date <= $3
                   ORDER BY local_date DESC""",
                user_id,
                today - timedelta(days=2),
                today,
            )
            low_sleep_days = sum(
                1 for r in rows if r.get("flags") and "low_sleep" in (r["flags"] or [])
            )
            if low_sleep_days >= 3:
                candidates.append(
                    (
                        70,
                        {
                            "kind": "wellbeing",
                            "title": "Noticed you've been running light on sleep",
                            "body": "Want me to keep today low-key and push non-urgent reminders to tomorrow?",
                            "payload": {"type": "low_sleep_streak", "days": low_sleep_days},
                        },
                    )
                )
        except Exception as e:
            logger.debug("low_sleep check failed: %s", e)

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]
