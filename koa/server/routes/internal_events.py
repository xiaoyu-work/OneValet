"""Internal calendar events CRUD — service-key protected.

Owns the events surface for koa's ``tenant_default.local_calendar_events``
table. Called by:

* the AI agent (via ``LocalCalendarProvider``, in-process — same DB handle),
* and (in PR2) the koi-backend gateway proxying iOS app requests.

Writability rules
-----------------
The table holds rows from three sources: ``eventkit`` (iOS EventKit
ingest), ``google`` (CalendarSyncService mirror), and ``local``
(AI-/app-created).

Local rows are mutable directly. Non-local rows (``google`` /
``eventkit``) follow the **two-way calendar sync** toggle stored in
``tenant_default.user_settings``:

* When the toggle is OFF (default), PATCH/DELETE on a non-local row
  returns 403 with a message pointing the user at Settings.
* When ON, the writeback service in
  ``koa/services/calendar_writeback.py`` propagates the change to the
  source provider (Google API for ``google`` rows; the iOS app handles
  EventKit on-device and we just patch the cache).

For the AI-agent codepath (``LocalCalendarProvider``) we still refuse
non-local edits — flipping a user-facing toggle should not implicitly
grant write permission to the AI.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...services import calendar_writeback as cwb
from ...services.user_settings import (
    KEY_TWO_WAY_CALENDAR_SYNC,
    get_user_setting,
    set_user_setting,
)
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
        "color": row.get("color"),
        "recurrence_rule": row.get("recurrence_rule"),
        "reminder_minutes": list(row["reminder_minutes"])
        if row.get("reminder_minutes") is not None
        else None,
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
        "       all_day, location, notes, attendees, metadata,",
        "       color, recurrence_rule, reminder_minutes,",
        "       source, updated_at",
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
    color: Optional[str] = None
    recurrence_rule: Optional[str] = None
    reminder_minutes: Optional[List[int]] = None
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
             all_day, location, notes, attendees, metadata,
             color, recurrence_rule, reminder_minutes,
             source, updated_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::jsonb,
                $12,$13,$14,'local',NOW())
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
            color = EXCLUDED.color,
            recurrence_rule = EXCLUDED.recurrence_rule,
            reminder_minutes = EXCLUDED.reminder_minutes,
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
        req.color,
        req.recurrence_rule,
        req.reminder_minutes,
    )

    row = await db.fetchrow(
        """
        SELECT user_id, event_id, calendar_name, title, starts_at, ends_at,
               all_day, location, notes, attendees, metadata,
               color, recurrence_rule, reminder_minutes,
               source, updated_at
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
    color: Optional[str] = None
    recurrence_rule: Optional[str] = None
    reminder_minutes: Optional[List[int]] = None


@router.patch(
    "/api/internal/events/{event_id}",
    dependencies=[Depends(verify_service_key)],
)
async def update_event(event_id: str, req: EventUpdate) -> Dict[str, Any]:
    app = require_app()
    db = _require_db(app)

    existing = await db.fetchrow(
        "SELECT user_id, event_id, calendar_name, title, starts_at, ends_at, "
        "all_day, location, notes, attendees, metadata, "
        "color, recurrence_rule, reminder_minutes, "
        "source, updated_at "
        "FROM tenant_default.local_calendar_events "
        "WHERE user_id = $1 AND event_id = $2",
        req.tenant_id,
        event_id,
    )
    if existing is None:
        raise HTTPException(404, "Event not found")

    if existing["source"] != "local":
        # Non-local rows go through the writeback service: gated on the
        # user's two-way-sync toggle and pushed to the source provider
        # (Google) or patched-locally + client-mirrored (EventKit).
        fields = req.model_dump(exclude={"tenant_id"}, exclude_none=True)
        try:
            row = await cwb.propagate_update(
                db=db,
                calendar_sync=app.calendar_sync,
                tenant_id=req.tenant_id,
                existing_row=dict(existing),
                fields=fields,
            )
        except cwb.ToggleOff as e:
            raise HTTPException(403, str(e))
        except cwb.UnsupportedSource as e:
            raise HTTPException(400, str(e))
        except cwb.CredentialsMissing as e:
            raise HTTPException(409, str(e))
        except cwb.SourceWriteFailed as e:
            raise HTTPException(502, f"Source calendar refused the change: {e}")
        return {"updated": True, "event": _row_to_event(dict(row))}

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
    if req.color is not None:
        add("color", req.color)
    if req.recurrence_rule is not None:
        add("recurrence_rule", req.recurrence_rule)
    if req.reminder_minutes is not None:
        add("reminder_minutes", req.reminder_minutes)

    if not sets:
        # Nothing to change — just return current row.
        return {"updated": False, "event": _row_to_event(dict(existing))}

    sets.append("updated_at = NOW()")
    args.append(req.tenant_id)
    args.append(event_id)
    sql = (
        "UPDATE tenant_default.local_calendar_events SET "
        + ", ".join(sets)
        + f" WHERE user_id = ${len(args) - 1} AND event_id = ${len(args)} "
        "RETURNING user_id, event_id, calendar_name, title, starts_at, ends_at, "
        "all_day, location, notes, attendees, metadata, "
        "color, recurrence_rule, reminder_minutes, "
        "source, updated_at"
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
        "SELECT user_id, event_id, calendar_name, title, starts_at, ends_at, "
        "all_day, location, notes, attendees, metadata, "
        "color, recurrence_rule, reminder_minutes, "
        "source, updated_at "
        "FROM tenant_default.local_calendar_events "
        "WHERE user_id = $1 AND event_id = $2",
        tenant_id,
        event_id,
    )
    if row is None:
        return {"deleted": False}

    if row["source"] != "local":
        try:
            await cwb.propagate_delete(
                db=db,
                tenant_id=tenant_id,
                existing_row=dict(row),
            )
        except cwb.ToggleOff as e:
            raise HTTPException(403, str(e))
        except cwb.UnsupportedSource as e:
            raise HTTPException(400, str(e))
        except cwb.CredentialsMissing as e:
            raise HTTPException(409, str(e))
        except cwb.SourceWriteFailed as e:
            raise HTTPException(502, f"Source calendar refused the change: {e}")
        return {"deleted": True}

    await db.execute(
        "DELETE FROM tenant_default.local_calendar_events WHERE user_id = $1 AND event_id = $2",
        tenant_id,
        event_id,
    )
    return {"deleted": True}


# ── Trigger calendar sync ────────────────────────────────────────────────


class CalendarSyncReq(BaseModel):
    tenant_id: str


@router.post(
    "/api/internal/calendar/sync",
    dependencies=[Depends(verify_service_key)],
)
async def trigger_calendar_sync(req: CalendarSyncReq) -> Dict[str, Any]:
    """Run a one-shot CalendarSync pass for ``tenant_id``.

    Service-key-protected sibling to ``/api/sensing/calendar/sync``
    (which uses the weaker X-API-Key auth). Called by koi-backend's
    Google/Outlook push-notification webhooks: instead of writing
    events into a now-defunct Supabase table, the gateway just nudges
    koa to re-poll the source calendar so its
    ``local_calendar_events`` table stays current.
    """
    app = require_app()
    svc = getattr(app, "calendar_sync", None)
    if svc is None:
        raise HTTPException(503, "CalendarSyncService not available")
    return await svc.sync_tenant(req.tenant_id)


# ── User settings ────────────────────────────────────────────────────────


class UserSettingPut(BaseModel):
    tenant_id: str
    value: Any


@router.get(
    "/api/internal/user-settings/{key}",
    dependencies=[Depends(verify_service_key)],
)
async def read_user_setting(
    key: str,
    tenant_id: str = Query(...),
) -> Dict[str, Any]:
    """Return the JSON value for a user setting key (or ``None``)."""
    db = _require_db(require_app())
    value = await get_user_setting(db, tenant_id, key, default=None)
    return {"key": key, "value": value}


@router.put(
    "/api/internal/user-settings/{key}",
    dependencies=[Depends(verify_service_key)],
)
async def write_user_setting(
    key: str,
    req: UserSettingPut,
) -> Dict[str, Any]:
    """Upsert a user setting; ``value`` is stored as JSON."""
    if key == KEY_TWO_WAY_CALENDAR_SYNC and not isinstance(req.value, bool):
        raise HTTPException(400, f"{key!r} must be boolean")
    db = _require_db(require_app())
    await set_user_setting(db, req.tenant_id, key, req.value)
    return {"key": key, "value": req.value}
