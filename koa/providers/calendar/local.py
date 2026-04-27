"""LocalCalendarProvider — direct DB-backed calendar provider.

Implements the calendar provider contract against koa's own
``tenant_default.local_calendar_events`` table without any HTTP hops.
The previous version called back into koi-backend, which was the wrong
direction (AI engine depending on app gateway) and meant agent could
never see Google-mirrored events that CalendarSyncService writes here.

Writability follows the source rule enforced in
``koa/server/routes/internal_events.py``: only rows with ``source='local'``
can be updated/deleted via the agent. Google-mirrored or EventKit-ingested
rows must be mutated on the source calendar; the agent will surface a
helpful error rather than trash its local cache.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _row_to_event(row: dict) -> Dict[str, Any]:
    """Map a DB row to the shape the calendar agent expects.

    Note: the agent's existing tools read both ``id``/``event_id``,
    ``summary``/``title``, and ``start``/``end`` (datetimes). We expose
    all variants for compatibility.
    """
    starts_at = row.get("starts_at")
    ends_at = row.get("ends_at")
    return {
        "id": row.get("event_id"),
        "event_id": row.get("event_id"),
        "summary": row.get("title") or "No title",
        "title": row.get("title") or "No title",
        "description": row.get("notes") or "",
        "start": starts_at,
        "end": ends_at,
        "starts_at": starts_at,
        "ends_at": ends_at,
        "location": row.get("location") or "",
        "all_day": bool(row.get("all_day")),
        "calendar_name": row.get("calendar_name"),
        "source": row.get("source", "local"),
    }


class LocalCalendarProvider:
    """Direct DB-backed implementation. Reads from all sources; writes only ``local``."""

    def __init__(self, tenant_id: str, db: Any):
        if db is None:
            raise ValueError("LocalCalendarProvider requires a database handle")
        self.tenant_id = tenant_id
        self._db = db

    async def ensure_valid_token(self, force_refresh: bool = False) -> bool:
        return True

    async def list_events(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 10,
        query: Optional[str] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sql = [
            "SELECT user_id, event_id, calendar_name, title, starts_at, ends_at,",
            "       all_day, location, notes, attendees, metadata, source, updated_at",
            "FROM tenant_default.local_calendar_events",
            "WHERE user_id = $1",
        ]
        args: List[Any] = [self.tenant_id]
        if time_min is not None:
            args.append(time_min)
            sql.append(f"AND ends_at >= ${len(args)}")
        if time_max is not None:
            args.append(time_max)
            sql.append(f"AND starts_at <= ${len(args)}")
        if query:
            args.append(f"%{query}%")
            sql.append(f"AND title ILIKE ${len(args)}")
        args.append(max_results)
        sql.append(f"ORDER BY starts_at ASC LIMIT ${len(args)}")

        try:
            rows = await self._db.fetch(" ".join(sql), *args)
            events = [_row_to_event(dict(r)) for r in rows]
            return {"success": True, "data": events, "count": len(events)}
        except Exception as e:
            logger.error(f"LocalCalendarProvider.list_events failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        event_id = f"local:{uuid.uuid4()}"
        attendees_json = json.dumps(attendees) if attendees is not None else None

        try:
            row = await self._db.fetchrow(
                """
                INSERT INTO tenant_default.local_calendar_events
                    (user_id, event_id, calendar_name, title, starts_at, ends_at,
                     all_day, location, notes, attendees, metadata, source, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,FALSE,$7,$8,$9::jsonb,NULL,'local',NOW())
                RETURNING user_id, event_id, calendar_name, title, starts_at, ends_at,
                          all_day, location, notes, attendees, metadata, source, updated_at
                """,
                self.tenant_id,
                event_id,
                calendar_id,
                summary,
                start,
                end,
                location,
                description,
                attendees_json,
            )
            event = _row_to_event(dict(row))
            return {"success": True, "event_id": event["event_id"], "data": event}
        except Exception as e:
            logger.error(f"LocalCalendarProvider.create_event failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def update_event(
        self,
        event_id: str,
        summary: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            existing = await self._db.fetchrow(
                "SELECT source FROM tenant_default.local_calendar_events "
                "WHERE user_id = $1 AND event_id = $2",
                self.tenant_id,
                event_id,
            )
            if existing is None:
                return {"success": False, "error": "Event not found"}
            if existing["source"] != "local":
                return {
                    "success": False,
                    "error": (
                        f"This event came from {existing['source']} and can't be edited "
                        "here. Update it on the source calendar instead."
                    ),
                }

            sets: List[str] = []
            args: List[Any] = []

            def add(col: str, val: Any, jsonb: bool = False) -> None:
                args.append(val)
                token = f"${len(args)}"
                if jsonb:
                    token += "::jsonb"
                sets.append(f"{col} = {token}")

            if summary is not None:
                add("title", summary)
            if start is not None:
                add("starts_at", start)
            if end is not None:
                add("ends_at", end)
            if description is not None:
                add("notes", description)
            if location is not None:
                add("location", location)
            if calendar_id is not None:
                add("calendar_name", calendar_id)
            if attendees is not None:
                add("attendees", json.dumps(attendees), jsonb=True)

            if not sets:
                row = await self._db.fetchrow(
                    "SELECT user_id, event_id, calendar_name, title, starts_at, ends_at, "
                    "all_day, location, notes, attendees, metadata, source, updated_at "
                    "FROM tenant_default.local_calendar_events "
                    "WHERE user_id = $1 AND event_id = $2",
                    self.tenant_id,
                    event_id,
                )
                event = _row_to_event(dict(row))
                return {"success": True, "event_id": event["event_id"], "data": event}

            sets.append("updated_at = NOW()")
            args.append(self.tenant_id)
            args.append(event_id)
            sql = (
                "UPDATE tenant_default.local_calendar_events SET "
                + ", ".join(sets)
                + f" WHERE user_id = ${len(args) - 1} AND event_id = ${len(args)} "
                "RETURNING user_id, event_id, calendar_name, title, starts_at, ends_at, "
                "all_day, location, notes, attendees, metadata, source, updated_at"
            )
            row = await self._db.fetchrow(sql, *args)
            event = _row_to_event(dict(row))
            return {"success": True, "event_id": event["event_id"], "data": event}
        except Exception as e:
            logger.error(f"LocalCalendarProvider.update_event failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            row = await self._db.fetchrow(
                "SELECT source FROM tenant_default.local_calendar_events "
                "WHERE user_id = $1 AND event_id = $2",
                self.tenant_id,
                event_id,
            )
            if row is None:
                return {"success": False, "error": "Event not found"}
            if row["source"] != "local":
                return {
                    "success": False,
                    "error": (
                        f"This event came from {row['source']} and can't be deleted "
                        "here. Delete it on the source calendar instead."
                    ),
                }
            await self._db.execute(
                "DELETE FROM tenant_default.local_calendar_events "
                "WHERE user_id = $1 AND event_id = $2",
                self.tenant_id,
                event_id,
            )
            return {"success": True}
        except Exception as e:
            logger.error(f"LocalCalendarProvider.delete_event failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
