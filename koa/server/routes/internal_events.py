"""Internal calendar events CRUD — service-key protected.

Owns the events surface for koa's ``tenant_default.local_calendar_events``
table. Called by:

* the AI agent (via ``LocalCalendarProvider``, in-process — same DB handle),
* and (in PR2) the koi-backend gateway proxying iOS app requests.

Writability rules
-----------------
The table holds rows from three sources: ``eventkit`` (iOS EventKit
ingest), ``google`` (CalendarSyncService mirror), and ``local``
(AI-/app-created). Only ``source='local'`` rows are mutable here —
PATCH/DELETE on a Google or EventKit row would just trash the cache;
the source calendar is unchanged and the row would reincarnate on the
next sync. We return 403 in that case.

For ``source='google'`` you should mutate via Google's API (not yet wired
on the agent side); for ``source='eventkit'`` you must mutate on the
phone via EventKit.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..app import require_app, verify_service_key

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_db(app):
    db = getattr(app, "database", None)
    if db is None:
        raise HTTPException(503, "Database not initialised")
    return db


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise HTTPException(400, f"Invalid datetime: {value}") from e
    raise HTTPException(400, f"Invalid datetime type: {type(value).__name__}")


def _row_to_event(row: dict) -> Dict[str, Any]:
    """Shape the DB row into a stable wire format."""
    return {
        "event_id": row["event_id"],
        "user_id": row["user_id"],
        "calendar_name": row.get("calendar_name"),
        "title": row.get("title"),
        "starts_at": row["starts_at"].isoformat() if row.get("starts_at") else None,
        "ends_at": row["ends_at"].isoformat() if row.get("ends_at") else None,
        "all_day": bool(row.get("all_day")),
        "location": row.get("location"),
        "notes": row.get("notes"),
        "attendees": json.loads(row["attendees"]) if row.get("attendees") else None,
        "metadata": json.loads(row["metadata"]) if row.get("metadata") else None,
        "source": row.get("source", "local"),
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


# ── List ─────────────────────────────────────────────────────────────────


@router.get("/api/internal/events", dependencies=[Depends(verify_service_key)])
async def list_events(
    tenant_id: str = Query(...),
    time_min: Optional[str] = Query(None),
    time_max: Optional[str] = Query(None),
    query: Optional[str] = Query(None, description="Substring match on title"),
    source: Optional[str] = Query(None, description="Filter by source"),
    max_results: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    db = _require_db(require_app())

    sql = [
        "SELECT user_id, event_id, calendar_name, title, starts_at, ends_at,",
        "       all_day, location, notes, attendees, metadata, source, updated_at",
        "FROM tenant_default.local_calendar_events",
        "WHERE user_id = $1",
    ]
    args: List[Any] = [tenant_id]

    if time_min is not None:
        args.append(_parse_ts(time_min))
        sql.append(f"AND ends_at >= ${len(args)}")
    if time_max is not None:
        args.append(_parse_ts(time_max))
        sql.append(f"AND starts_at <= ${len(args)}")
    if query:
        args.append(f"%{query}%")
        sql.append(f"AND title ILIKE ${len(args)}")
    if source:
        args.append(source)
        sql.append(f"AND source = ${len(args)}")

    args.append(max_results)
    sql.append(f"ORDER BY starts_at ASC LIMIT ${len(args)}")

    rows = await db.fetch(" ".join(sql), *args)
    return {"events": [_row_to_event(dict(r)) for r in rows]}


# ── Create ───────────────────────────────────────────────────────────────


class EventCreate(BaseModel):
    tenant_id: str
    title: str
    start_at: str
    end_at: str
    all_day: bool = False
    description: Optional[str] = None
    location: Optional[str] = None
    calendar_name: Optional[str] = None
    attendees: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    event_id: Optional[str] = Field(
        None,
        description="If omitted, a 'local:<uuid>' id is generated. Always written with source='local'.",
    )


@router.post("/api/internal/events", dependencies=[Depends(verify_service_key)])
async def create_event(req: EventCreate) -> Dict[str, Any]:
    db = _require_db(require_app())

    starts_at = _parse_ts(req.start_at)
    ends_at = _parse_ts(req.end_at)
    if starts_at is None or ends_at is None:
        raise HTTPException(400, "start_at and end_at are required")
    if ends_at < starts_at:
        raise HTTPException(400, "end_at must be >= start_at")

    event_id = req.event_id or f"local:{uuid.uuid4()}"

    await db.execute(
        """
        INSERT INTO tenant_default.local_calendar_events
            (user_id, event_id, calendar_name, title, starts_at, ends_at,
             all_day, location, notes, attendees, metadata, source, updated_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::jsonb,'local',NOW())
        ON CONFLICT (user_id, event_id) DO UPDATE SET
            calendar_name = EXCLUDED.calendar_name,
            title = EXCLUDED.title,
            starts_at = EXCLUDED.starts_at,
            ends_at = EXCLUDED.ends_at,
            all_day = EXCLUDED.all_day,
            location = EXCLUDED.location,
            notes = EXCLUDED.notes,
            attendees = EXCLUDED.attendees,
            metadata = EXCLUDED.metadata,
            source = 'local',
            updated_at = NOW()
        """,
        req.tenant_id,
        event_id,
        req.calendar_name,
        req.title,
        starts_at,
        ends_at,
        bool(req.all_day),
        req.location,
        req.description,
        json.dumps(req.attendees) if req.attendees is not None else None,
        json.dumps(req.metadata) if req.metadata is not None else None,
    )

    row = await db.fetchrow(
        """
        SELECT user_id, event_id, calendar_name, title, starts_at, ends_at,
               all_day, location, notes, attendees, metadata, source, updated_at
        FROM tenant_default.local_calendar_events
        WHERE user_id = $1 AND event_id = $2
        """,
        req.tenant_id,
        event_id,
    )
    return {"created": True, "event": _row_to_event(dict(row))}


# ── Update ───────────────────────────────────────────────────────────────


class EventUpdate(BaseModel):
    tenant_id: str
    title: Optional[str] = None
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    all_day: Optional[bool] = None
    description: Optional[str] = None
    location: Optional[str] = None
    calendar_name: Optional[str] = None
    attendees: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@router.patch(
    "/api/internal/events/{event_id}",
    dependencies=[Depends(verify_service_key)],
)
async def update_event(event_id: str, req: EventUpdate) -> Dict[str, Any]:
    db = _require_db(require_app())

    existing = await db.fetchrow(
        "SELECT source FROM tenant_default.local_calendar_events "
        "WHERE user_id = $1 AND event_id = $2",
        req.tenant_id,
        event_id,
    )
    if existing is None:
        raise HTTPException(404, "Event not found")
    if existing["source"] != "local":
        raise HTTPException(
            403,
            f"Event source is '{existing['source']}', not 'local' — "
            "mutate on the source calendar instead",
        )

    sets: List[str] = []
    args: List[Any] = []

    def add(col: str, val: Any) -> None:
        args.append(val)
        sets.append(f"{col} = ${len(args)}")

    if req.title is not None:
        add("title", req.title)
    if req.start_at is not None:
        add("starts_at", _parse_ts(req.start_at))
    if req.end_at is not None:
        add("ends_at", _parse_ts(req.end_at))
    if req.all_day is not None:
        add("all_day", bool(req.all_day))
    if req.description is not None:
        add("notes", req.description)
    if req.location is not None:
        add("location", req.location)
    if req.calendar_name is not None:
        add("calendar_name", req.calendar_name)
    if req.attendees is not None:
        add("attendees", json.dumps(req.attendees))
        sets[-1] = sets[-1] + "::jsonb"
    if req.metadata is not None:
        add("metadata", json.dumps(req.metadata))
        sets[-1] = sets[-1] + "::jsonb"

    if not sets:
        # Nothing to change — just return current row.
        row = await db.fetchrow(
            "SELECT user_id, event_id, calendar_name, title, starts_at, ends_at, "
            "all_day, location, notes, attendees, metadata, source, updated_at "
            "FROM tenant_default.local_calendar_events "
            "WHERE user_id = $1 AND event_id = $2",
            req.tenant_id,
            event_id,
        )
        return {"updated": False, "event": _row_to_event(dict(row))}

    sets.append("updated_at = NOW()")
    args.append(req.tenant_id)
    args.append(event_id)
    sql = (
        "UPDATE tenant_default.local_calendar_events SET "
        + ", ".join(sets)
        + f" WHERE user_id = ${len(args) - 1} AND event_id = ${len(args)} "
        "RETURNING user_id, event_id, calendar_name, title, starts_at, ends_at, "
        "all_day, location, notes, attendees, metadata, source, updated_at"
    )
    row = await db.fetchrow(sql, *args)
    return {"updated": True, "event": _row_to_event(dict(row))}


# ── Delete ───────────────────────────────────────────────────────────────


@router.delete(
    "/api/internal/events/{event_id}",
    dependencies=[Depends(verify_service_key)],
)
async def delete_event(
    event_id: str,
    tenant_id: str = Query(...),
) -> Dict[str, Any]:
    db = _require_db(require_app())

    row = await db.fetchrow(
        "SELECT source FROM tenant_default.local_calendar_events "
        "WHERE user_id = $1 AND event_id = $2",
        tenant_id,
        event_id,
    )
    if row is None:
        return {"deleted": False}
    if row["source"] != "local":
        raise HTTPException(
            403,
            f"Event source is '{row['source']}', not 'local' — "
            "delete on the source calendar instead",
        )

    await db.execute(
        "DELETE FROM tenant_default.local_calendar_events "
        "WHERE user_id = $1 AND event_id = $2",
        tenant_id,
        event_id,
    )
    return {"deleted": True}
