"""EventKit adapter for CalendarAgent.

Unified calendar view: Google/iCloud via Composio *plus* local iOS EventKit
events (ingested by koi-backend route /api/eventkit/sync into
``local_calendar_events``).

Phase-3 stub: signatures + contracts pinned; the CalendarAgent's LLM tool
layer can import `fetch_upcoming_local(db, user_id, window)` and merge the
result with Composio data before constructing the prompt context.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


async def fetch_upcoming_local(
    db,
    user_id: str,
    *,
    hours_ahead: int = 48,
    calendars: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return local EventKit events in ``[now, now+hours_ahead]``.

    The shape mirrors Composio calendar events for easy merge:
      { source: 'eventkit', id, title, start, end, location, calendar, all_day }
    """
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)
    q = """SELECT event_id, title, starts_at, ends_at, location, calendar_name, all_day
           FROM local_calendar_events
           WHERE user_id = $1 AND starts_at <= $3 AND ends_at >= $2"""
    args: List[Any] = [user_id, now, until]
    if calendars:
        q += " AND calendar_name = ANY($4)"
        args.append(calendars)
    q += " ORDER BY starts_at ASC LIMIT 100"
    rows = await db.fetch(q, *args)
    return [
        {
            "source": "eventkit",
            "id": r["event_id"],
            "title": r["title"],
            "start": r["starts_at"].isoformat() if r["starts_at"] else None,
            "end": r["ends_at"].isoformat() if r["ends_at"] else None,
            "location": r.get("location"),
            "calendar": r.get("calendar_name"),
            "all_day": bool(r.get("all_day")),
        }
        for r in rows
    ]


# TODO(phase-3): plug `fetch_upcoming_local` into CalendarAgent.agent so that
# prompt context contains a unified list.  Ensure dedupe by (title, start)
# when the same event is mirrored to both Google and iCloud.
