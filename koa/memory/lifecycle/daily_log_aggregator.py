"""DailyLogAggregator — builds one ``daily_logs`` row per user per local day.

This is a *pure-SQL* step: no LLM.  It rolls up the day's messages, tool
calls, calendar events, user_state, health samples, and sensing output into
a single compact JSONB blob that the WeeklyReflector will consume later.

Running this nightly means the weekly reflection only needs to read 7 rows
per user (one per day), not thousands of messages.  That's how we keep
weekly reflection cost bounded regardless of engagement depth.

Cron suggestion: 02:00 in the user's local timezone, processes yesterday.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def aggregate_day(db, user_id: str, local_date: date, tz_name: str) -> Dict[str, Any]:
    """Build and upsert the daily_logs row for ``local_date``.  Returns the
    aggregated dict (also what was written)."""
    start_utc, end_utc = _local_day_bounds(local_date, tz_name)

    messages = await _message_summary(db, user_id, start_utc, end_utc)
    tools = await _tool_summary(db, user_id, start_utc, end_utc)
    calendar = await _calendar_summary(db, user_id, start_utc, end_utc)
    reminders = await _reminder_summary(db, user_id, start_utc, end_utc)
    state = await _state_summary(db, user_id, local_date)
    top_entities = await _top_entities(db, user_id, start_utc, end_utc)

    payload: Dict[str, Any] = {
        "local_date": local_date.isoformat(),
        "timezone": tz_name,
        "messages": messages,
        "tools": tools,
        "calendar": calendar,
        "reminders": reminders,
        "state": state,
        "top_entities": top_entities,
    }

    await _upsert_daily_log(db, user_id, local_date, payload)
    return payload


async def _message_summary(db, user_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    try:
        rows = await db.fetch(
            """SELECT role, COUNT(*) AS c,
                      COALESCE(SUM(LENGTH(content)), 0) AS chars
               FROM messages
               WHERE user_id = $1 AND created_at >= $2 AND created_at < $3
               GROUP BY role""",
            user_id, start, end,
        )
        by_role = {r["role"]: {"count": r["c"], "chars": r["chars"]} for r in rows}
        total = sum(r["c"] for r in rows)
        return {"total": total, "by_role": by_role}
    except Exception as e:
        logger.debug("_message_summary failed: %s", e)
        return {"total": 0, "by_role": {}}


async def _tool_summary(db, user_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    """Approximate tool usage via agent_invocations (existing table)."""
    try:
        rows = await db.fetch(
            """SELECT agent_name, COUNT(*) AS c
               FROM agent_invocations
               WHERE user_id = $1 AND created_at >= $2 AND created_at < $3
               GROUP BY agent_name
               ORDER BY c DESC LIMIT 20""",
            user_id, start, end,
        )
        return {r["agent_name"]: r["c"] for r in rows}
    except Exception as e:
        logger.debug("_tool_summary failed (table may not exist): %s", e)
        return {}


async def _calendar_summary(db, user_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    """Combine internal_calendar + local_calendar_events for the day."""
    out: Dict[str, Any] = {"events": [], "count": 0}
    try:
        internal = await db.fetch(
            """SELECT title, starts_at, ends_at FROM internal_calendar
               WHERE user_id = $1 AND starts_at >= $2 AND starts_at < $3
               ORDER BY starts_at LIMIT 50""",
            user_id, start, end,
        )
        out["events"].extend([{"title": r["title"], "starts_at": r["starts_at"].isoformat(), "source": "internal"} for r in internal])
    except Exception as e:
        logger.debug("internal_calendar query skipped: %s", e)

    try:
        local = await db.fetch(
            """SELECT title, starts_at FROM local_calendar_events
               WHERE user_id = $1 AND starts_at >= $2 AND starts_at < $3
               ORDER BY starts_at LIMIT 50""",
            user_id, start, end,
        )
        out["events"].extend([{"title": r["title"], "starts_at": r["starts_at"].isoformat(), "source": "eventkit"} for r in local])
    except Exception as e:
        logger.debug("local_calendar_events query skipped: %s", e)

    out["count"] = len(out["events"])
    return out


async def _reminder_summary(db, user_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    try:
        rows = await db.fetch(
            """SELECT title, completed FROM local_reminders
               WHERE user_id = $1 AND (
                   (completed AND completed_at >= $2 AND completed_at < $3)
                   OR (due_at >= $2 AND due_at < $3)
               )
               LIMIT 50""",
            user_id, start, end,
        )
        completed = [r["title"] for r in rows if r["completed"]]
        due = [r["title"] for r in rows if not r["completed"]]
        return {"completed": completed, "due": due}
    except Exception as e:
        logger.debug("_reminder_summary failed: %s", e)
        return {}


async def _state_summary(db, user_id: str, local_date: date) -> Dict[str, Any]:
    try:
        row = await db.fetchrow(
            """SELECT sleep_minutes, sleep_score, steps, activity_minutes,
                      stress_score, mood, primary_location, flags, source_data
               FROM user_state WHERE user_id = $1 AND local_date = $2""",
            user_id, local_date,
        )
        return dict(row) if row else {}
    except Exception as e:
        logger.debug("_state_summary failed: %s", e)
        return {}


async def _top_entities(db, user_id: str, start: datetime, end: datetime) -> List[str]:
    """Heuristic: pick top message mentions.  For now we rely on the weekly
    reflector to do NER; here we just forward an empty list."""
    # TODO: once an entity extractor lands, populate this.
    return []


async def _upsert_daily_log(db, user_id: str, local_date: date, payload: Dict[str, Any]):
    try:
        await db.execute(
            """INSERT INTO daily_logs (user_id, local_date, payload)
               VALUES ($1, $2, $3::jsonb)
               ON CONFLICT (user_id, local_date)
               DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()""",
            user_id, local_date, json.dumps(payload, default=str),
        )
    except Exception as e:
        logger.error("daily_logs upsert failed: %s", e)


def _local_day_bounds(local_date: date, tz_name: str):
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name) if tz_name and tz_name != "UTC" else timezone.utc
    except Exception:
        tz = timezone.utc
    start_local = datetime.combine(local_date, datetime.min.time(), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
