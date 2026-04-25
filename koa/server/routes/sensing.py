"""Sensing ingest endpoints.

Called by koi-backend (thin proxy) which authenticates the iOS user and
injects ``user_id``. Service-token protected so only our own backend can hit
these.

Writes raw sensor rows into the ``tenant_default`` schema (migration 011).
Upserts are idempotent per (user_id, natural key) so retries + backfill are
safe.

Aggregation / reflection does **not** happen here — that runs in the
reflection agents on a cron. This endpoint just persists.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...errors import E, KoaError
from ..app import require_app, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_db(app):
    db = getattr(app, "database", None)
    if db is None:
        raise KoaError(E.SERVICE_UNAVAILABLE, "Database not initialised", details={"service": "db"})
    return db


def _parse_ts(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    if isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _jsonify(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return json.dumps(v, default=str)
    except TypeError:
        return None


# ---------------------------------------------------------------- Models


class _BaseIngest(BaseModel):
    user_id: str


class HealthSample(BaseModel):
    type: str
    started_at: Any
    ended_at: Optional[Any] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


class HealthKitIngest(_BaseIngest):
    samples: List[HealthSample] = Field(default_factory=list)


class MotionSegment(BaseModel):
    started_at: Any
    ended_at: Any
    activity: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class MotionIngest(_BaseIngest):
    segments: List[MotionSegment] = Field(default_factory=list)


class DeviceContact(BaseModel):
    contact_hash: str
    display_name: Optional[str] = None
    channel_hints: Optional[Dict[str, Any]] = None
    last_interaction_at: Optional[Any] = None
    interaction_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class ContactsIngest(_BaseIngest):
    contacts: List[DeviceContact] = Field(default_factory=list)


class CalendarEvent(BaseModel):
    event_id: str
    calendar_name: Optional[str] = None
    title: Optional[str] = None
    starts_at: Optional[Any] = None
    ends_at: Optional[Any] = None
    all_day: bool = False
    location: Optional[str] = None
    notes: Optional[str] = None
    attendees: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class Reminder(BaseModel):
    reminder_id: str
    list_name: Optional[str] = None
    title: Optional[str] = None
    notes: Optional[str] = None
    due_at: Optional[Any] = None
    completed: bool = False
    completed_at: Optional[Any] = None
    priority: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class EventKitIngest(_BaseIngest):
    events: List[CalendarEvent] = Field(default_factory=list)
    reminders: List[Reminder] = Field(default_factory=list)


# ---------------------------------------------------------------- Endpoints


@router.post("/api/sensing/healthkit", dependencies=[Depends(verify_api_key)])
async def ingest_healthkit(req: HealthKitIngest) -> Dict[str, Any]:
    if not req.samples:
        return {"written": 0}
    db = _require_db(require_app())
    written = 0
    for s in req.samples:
        start = _parse_ts(s.started_at)
        if start is None:
            continue
        await db.execute(
            """
            INSERT INTO tenant_default.health_samples
                (user_id, type, started_at, ended_at, value, unit, metadata, source)
            VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8)
            ON CONFLICT (user_id, type, started_at) DO UPDATE SET
                ended_at = EXCLUDED.ended_at,
                value = EXCLUDED.value,
                unit = EXCLUDED.unit,
                metadata = EXCLUDED.metadata,
                source = EXCLUDED.source
            """,
            req.user_id,
            s.type,
            start,
            _parse_ts(s.ended_at),
            s.value,
            s.unit,
            _jsonify(s.metadata),
            s.source,
        )
        written += 1
    return {"written": written}


@router.post("/api/sensing/motion", dependencies=[Depends(verify_api_key)])
async def ingest_motion(req: MotionIngest) -> Dict[str, Any]:
    if not req.segments:
        return {"written": 0}
    db = _require_db(require_app())
    written = 0
    for seg in req.segments:
        start = _parse_ts(seg.started_at)
        end = _parse_ts(seg.ended_at)
        if start is None or end is None:
            continue
        await db.execute(
            """
            INSERT INTO tenant_default.motion_segments
                (user_id, started_at, ended_at, activity, confidence, metadata)
            VALUES ($1,$2,$3,$4,$5,$6::jsonb)
            ON CONFLICT (user_id, started_at, activity) DO UPDATE SET
                ended_at = EXCLUDED.ended_at,
                confidence = EXCLUDED.confidence,
                metadata = EXCLUDED.metadata
            """,
            req.user_id,
            start,
            end,
            seg.activity,
            seg.confidence,
            _jsonify(seg.metadata),
        )
        written += 1
    return {"written": written}


@router.post("/api/sensing/contacts", dependencies=[Depends(verify_api_key)])
async def ingest_contacts(req: ContactsIngest) -> Dict[str, Any]:
    if not req.contacts:
        return {"written": 0}
    db = _require_db(require_app())
    written = 0
    for c in req.contacts:
        await db.execute(
            """
            INSERT INTO tenant_default.device_contacts
                (user_id, contact_hash, display_name, channel_hints,
                 last_interaction_at, interaction_count, metadata, updated_at)
            VALUES ($1,$2,$3,$4::jsonb,$5,$6,$7::jsonb, NOW())
            ON CONFLICT (user_id, contact_hash) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                channel_hints = EXCLUDED.channel_hints,
                last_interaction_at = EXCLUDED.last_interaction_at,
                interaction_count = EXCLUDED.interaction_count,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            req.user_id,
            c.contact_hash,
            c.display_name,
            _jsonify(c.channel_hints),
            _parse_ts(c.last_interaction_at),
            int(c.interaction_count or 0),
            _jsonify(c.metadata),
        )
        written += 1
    return {"written": written}


@router.post("/api/sensing/eventkit", dependencies=[Depends(verify_api_key)])
async def ingest_eventkit(req: EventKitIngest) -> Dict[str, Any]:
    db = _require_db(require_app())
    ev_written = 0
    for ev in req.events:
        await db.execute(
            """
            INSERT INTO tenant_default.local_calendar_events
                (user_id, event_id, calendar_name, title, starts_at, ends_at,
                 all_day, location, notes, attendees, metadata, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::jsonb, NOW())
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
                updated_at = NOW()
            """,
            req.user_id,
            ev.event_id,
            ev.calendar_name,
            ev.title,
            _parse_ts(ev.starts_at),
            _parse_ts(ev.ends_at),
            bool(ev.all_day),
            ev.location,
            ev.notes,
            _jsonify(ev.attendees),
            _jsonify(ev.metadata),
        )
        ev_written += 1

    rem_written = 0
    for r in req.reminders:
        await db.execute(
            """
            INSERT INTO tenant_default.local_reminders
                (user_id, reminder_id, list_name, title, notes, due_at,
                 completed, completed_at, priority, metadata, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb, NOW())
            ON CONFLICT (user_id, reminder_id) DO UPDATE SET
                list_name = EXCLUDED.list_name,
                title = EXCLUDED.title,
                notes = EXCLUDED.notes,
                due_at = EXCLUDED.due_at,
                completed = EXCLUDED.completed,
                completed_at = EXCLUDED.completed_at,
                priority = EXCLUDED.priority,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            req.user_id,
            r.reminder_id,
            r.list_name,
            r.title,
            r.notes,
            _parse_ts(r.due_at),
            bool(r.completed),
            _parse_ts(r.completed_at),
            r.priority,
            _jsonify(r.metadata),
        )
        rem_written += 1

    return {"events_written": ev_written, "reminders_written": rem_written}


# ---------------------------------------------------------------- Read

_SUMMARY_ACCUM = {
    # sample_type → (field_in_response, op)   op: "sum_value_int" | "sum_minutes" | "avg"
    "steps": ("steps", "sum_value_int"),
    "active_energy": ("active_energy_kcal", "sum_value_int"),
    "sleep": ("sleep_minutes", "sum_minutes"),
    "mindful": ("mindful_minutes", "sum_minutes"),
    "workout": ("workout_minutes", "sum_minutes"),
    "hrv": ("hrv_avg_ms", "avg"),
    "resting_hr": ("resting_hr", "avg"),
    "heart_rate": ("heart_rate_avg", "avg"),
}


@router.get("/api/sensing/healthkit/summary", dependencies=[Depends(verify_api_key)])
async def health_daily_summary(
    user_id: str,
    date: Optional[str] = None,
):
    """Return a single-day health summary computed on the fly from
    ``health_samples``. ``date`` is YYYY-MM-DD (UTC); defaults to today UTC."""
    from datetime import date as _date
    from datetime import timedelta

    db = _require_db(require_app())
    if date:
        try:
            d = _date.fromisoformat(date)
        except ValueError:
            raise KoaError(E.VALIDATION, "date must be YYYY-MM-DD")
    else:
        d = datetime.now(timezone.utc).date()
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    rows = await db.fetch(
        """SELECT type, value, started_at, ended_at
           FROM tenant_default.health_samples
           WHERE user_id = $1 AND started_at >= $2 AND started_at < $3""",
        user_id,
        start,
        end,
    )

    out: Dict[str, Any] = {
        "date": d.isoformat(),
        "steps": 0,
        "active_energy_kcal": 0,
        "sleep_minutes": 0,
        "mindful_minutes": 0,
        "workout_minutes": 0,
        "hrv_avg_ms": None,
        "resting_hr": None,
        "heart_rate_avg": None,
        "mood": None,
        "sample_count": len(rows),
    }
    avg_buckets: Dict[str, List[float]] = {}
    for r in rows:
        t = r["type"]
        spec = _SUMMARY_ACCUM.get(t)
        if not spec:
            continue
        field, op = spec
        v = r["value"]
        if op == "sum_value_int" and v is not None:
            out[field] += int(v)
        elif op == "sum_minutes":
            # prefer duration from started_at/ended_at; fallback to value
            if r["ended_at"] and r["started_at"]:
                secs = (r["ended_at"] - r["started_at"]).total_seconds()
                out[field] += int(secs // 60)
            elif v is not None:
                out[field] += int(v)
        elif op == "avg" and v is not None:
            avg_buckets.setdefault(field, []).append(float(v))
    for field, xs in avg_buckets.items():
        if xs:
            out[field] = round(sum(xs) / len(xs), 2)

    try:
        st = await db.fetchrow(
            "SELECT mood FROM tenant_default.user_state WHERE user_id = $1 AND local_date = $2",
            user_id,
            d,
        )
        if st and st["mood"]:
            out["mood"] = st["mood"]
    except Exception as e:
        logger.debug("user_state mood lookup failed: %s", e)

    return out


@router.get("/api/sensing/healthkit/state", dependencies=[Depends(verify_api_key)])
async def health_recent_state(user_id: str, days: int = 7):
    from datetime import timedelta

    if days < 1 or days > 90:
        raise KoaError(E.VALIDATION, "days must be 1..90")
    db = _require_db(require_app())
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)
    try:
        rows = await db.fetch(
            """SELECT local_date, sleep_minutes, sleep_score, hrv_ms, resting_hr,
                      steps, activity_minutes, stress_score, mood, primary_location,
                      focus_mode, flags
               FROM tenant_default.user_state
               WHERE user_id = $1 AND local_date >= $2 AND local_date <= $3
               ORDER BY local_date""",
            user_id,
            start,
            today,
        )
    except Exception as e:
        logger.debug("user_state fetch failed: %s", e)
        rows = []
    return {
        "days": days,
        "state": [{**dict(r), "local_date": r["local_date"].isoformat()} for r in rows],
    }
