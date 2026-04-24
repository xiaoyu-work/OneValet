"""DailyLogAggregator — builds a daily_log *episode* per user per local day.

This is a *pure-SQL* rollup (no LLM). It reads the day's messages, tool
calls, calendar events, user_state, health samples and motion segments,
produces a compact JSON-shaped summary, and writes that summary to Momex
as an episode with ``subkind="daily_log"`` via :class:`EpisodeMemory`.

Running this nightly means the weekly reflector only needs to pull ~7
episodes per user (one per day) to see the whole week, instead of scanning
raw sensor rows.

Cron suggestion: 02:00 in the user's local timezone, processes yesterday.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


async def aggregate_day(
    db,
    user_id: str,
    local_date: date,
    tz_name: str,
    *,
    episode_memory=None,
) -> Dict[str, Any]:
    """Build the daily rollup and, if ``episode_memory`` is provided, write
    it to Momex as a daily_log episode. Returns the raw payload regardless.
    """
    start_utc, end_utc = _local_day_bounds(local_date, tz_name)

    messages = await _message_summary(db, user_id, start_utc, end_utc)
    tools = await _tool_summary(db, user_id, start_utc, end_utc)
    calendar = await _calendar_summary(db, user_id, start_utc, end_utc)
    reminders = await _reminder_summary(db, user_id, start_utc, end_utc)
    health = await _health_summary(db, user_id, start_utc, end_utc)
    motion = await _motion_summary(db, user_id, start_utc, end_utc)
    state = await _state_summary(db, user_id, local_date)

    payload: Dict[str, Any] = {
        "local_date": local_date.isoformat(),
        "timezone": tz_name,
        "messages": messages,
        "tools": tools,
        "calendar": calendar,
        "reminders": reminders,
        "health": health,
        "motion": motion,
        "state": state,
    }

    if episode_memory is not None:
        summary_text = _render_summary(payload)
        try:
            await episode_memory.write_episode(
                tenant_id=user_id,
                summary=summary_text,
                subkind="daily_log",
                start_ts=start_utc,
                end_ts=end_utc,
                source="sensing",
                extras={
                    "local_date": local_date.isoformat(),
                    "timezone": tz_name,
                    "payload": payload,
                },
            )
        except Exception as e:
            logger.error("daily_log episode write failed: %s", e)

    return payload


async def _message_summary(db, user_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    try:
        rows = await db.fetch(
            """SELECT role, COUNT(*) AS c,
                      COALESCE(SUM(LENGTH(content)), 0) AS chars
               FROM messages
               WHERE user_id = $1 AND created_at >= $2 AND created_at < $3
               GROUP BY role""",
            user_id,
            start,
            end,
        )
        by_role = {r["role"]: {"count": r["c"], "chars": r["chars"]} for r in rows}
        total = sum(r["c"] for r in rows)
        return {"total": total, "by_role": by_role}
    except Exception as e:
        logger.debug("_message_summary failed: %s", e)
        return {"total": 0, "by_role": {}}


async def _tool_summary(db, user_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    try:
        rows = await db.fetch(
            """SELECT agent_name, COUNT(*) AS c
               FROM agent_invocations
               WHERE user_id = $1 AND created_at >= $2 AND created_at < $3
               GROUP BY agent_name
               ORDER BY c DESC LIMIT 20""",
            user_id,
            start,
            end,
        )
        return {r["agent_name"]: r["c"] for r in rows}
    except Exception as e:
        logger.debug("_tool_summary failed (table may not exist): %s", e)
        return {}


async def _calendar_summary(db, user_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    out: Dict[str, Any] = {"events": [], "count": 0}
    try:
        internal = await db.fetch(
            """SELECT title, starts_at FROM internal_calendar
               WHERE user_id = $1 AND starts_at >= $2 AND starts_at < $3
               ORDER BY starts_at LIMIT 50""",
            user_id,
            start,
            end,
        )
        out["events"].extend(
            [
                {"title": r["title"], "starts_at": r["starts_at"].isoformat(), "source": "internal"}
                for r in internal
            ]
        )
    except Exception as e:
        logger.debug("internal_calendar query skipped: %s", e)

    try:
        local = await db.fetch(
            """SELECT title, starts_at FROM local_calendar_events
               WHERE user_id = $1 AND starts_at >= $2 AND starts_at < $3
               ORDER BY starts_at LIMIT 50""",
            user_id,
            start,
            end,
        )
        out["events"].extend(
            [
                {"title": r["title"], "starts_at": r["starts_at"].isoformat(), "source": "eventkit"}
                for r in local
            ]
        )
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
            user_id,
            start,
            end,
        )
        completed = [r["title"] for r in rows if r["completed"]]
        due = [r["title"] for r in rows if not r["completed"]]
        return {"completed": completed, "due": due}
    except Exception as e:
        logger.debug("_reminder_summary failed: %s", e)
        return {}


async def _health_summary(db, user_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    try:
        rows = await db.fetch(
            """SELECT type,
                      COUNT(*) AS c,
                      COALESCE(SUM(value), 0) AS total,
                      AVG(value) AS avg
               FROM health_samples
               WHERE user_id = $1 AND started_at >= $2 AND started_at < $3
               GROUP BY type""",
            user_id,
            start,
            end,
        )
        return {
            r["type"]: {
                "count": r["c"],
                "total": float(r["total"]) if r["total"] is not None else 0.0,
                "avg": float(r["avg"]) if r["avg"] is not None else None,
            }
            for r in rows
        }
    except Exception as e:
        logger.debug("_health_summary failed: %s", e)
        return {}


async def _motion_summary(db, user_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
    try:
        rows = await db.fetch(
            """SELECT activity,
                      SUM(EXTRACT(EPOCH FROM (ended_at - started_at))) AS seconds
               FROM motion_segments
               WHERE user_id = $1 AND started_at >= $2 AND started_at < $3
               GROUP BY activity""",
            user_id,
            start,
            end,
        )
        return {r["activity"]: int(float(r["seconds"] or 0)) for r in rows}
    except Exception as e:
        logger.debug("_motion_summary failed: %s", e)
        return {}


async def _state_summary(db, user_id: str, local_date: date) -> Dict[str, Any]:
    try:
        row = await db.fetchrow(
            """SELECT sleep_minutes, sleep_score, steps, activity_minutes,
                      stress_score, mood, primary_location, flags, source_data
               FROM user_state WHERE user_id = $1 AND local_date = $2""",
            user_id,
            local_date,
        )
        return dict(row) if row else {}
    except Exception as e:
        logger.debug("_state_summary failed: %s", e)
        return {}


def _render_summary(payload: Dict[str, Any]) -> str:
    """Human-readable one-paragraph summary used as the episode text
    (what Momex indexes for vector search)."""
    parts: List[str] = []
    local_date = payload.get("local_date", "")
    parts.append(f"Daily log {local_date}")

    state = payload.get("state") or {}
    if state:
        bits = []
        if state.get("sleep_minutes"):
            bits.append(f"sleep {int(state['sleep_minutes'])}m")
        if state.get("sleep_score") is not None:
            bits.append(f"sleep score {state['sleep_score']}")
        if state.get("steps"):
            bits.append(f"{int(state['steps'])} steps")
        if state.get("mood"):
            bits.append(f"mood={state['mood']}")
        if state.get("primary_location"):
            bits.append(f"@{state['primary_location']}")
        if bits:
            parts.append("; ".join(bits))

    cal = payload.get("calendar") or {}
    events = cal.get("events") or []
    if events:
        titles = [e.get("title") or "" for e in events[:5] if e.get("title")]
        if titles:
            parts.append("events: " + ", ".join(titles))

    rem = payload.get("reminders") or {}
    if rem.get("completed"):
        parts.append("done: " + ", ".join(rem["completed"][:5]))

    msgs = payload.get("messages") or {}
    if msgs.get("total"):
        parts.append(f"{msgs['total']} messages")

    return ". ".join(parts)


async def _upsert_daily_log(*args, **kwargs):  # back-compat shim (no-op)
    """Retained for any external caller; the daily_logs table no longer
    exists — the rollup is stored as a Momex episode now."""
    return None


def _local_day_bounds(local_date: date, tz_name: str):
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name) if tz_name and tz_name != "UTC" else timezone.utc
    except Exception:
        tz = timezone.utc
    start_local = datetime.combine(local_date, datetime.min.time(), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
