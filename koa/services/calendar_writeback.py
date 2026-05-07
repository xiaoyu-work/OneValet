"""Calendar writeback — propagate iOS-app edits/deletes to the source provider.

Triggered when the iOS app edits a row in ``local_calendar_events`` whose
``source != 'local'`` (today: ``google``, ``eventkit``). Gated by the
per-user ``two_way_calendar_sync`` setting.

This module owns the *full* mutation flow for non-local rows so the two
callsites that need it — ``server/routes/internal_events.py`` PATCH/DELETE
— don't drift:

1. Look up existing row.
2. Check the user's two-way-sync toggle. If OFF → ``ToggleOff``.
3. For ``source='google'``:
   * Resolve credentials via ``CalendarAccountResolver`` using
     ``metadata.account_name``.
   * Build a ``GoogleCalendarProvider`` and call ``update_event`` /
     ``delete_event`` on it.
   * On success, re-fetch the event from Google and re-upsert through
     ``CalendarSyncService._upsert_event`` so the local cache reflects
     authoritative state (instead of the request payload).
4. For ``source='eventkit'``:
   * Backend cannot reach iOS Calendar.app. Just mutate the local cache
     row to match the request — the iOS client is responsible for the
     device-side write via ``EventKitModule.updateEvent``.
5. Anything else → ``UnsupportedSource``.

AI-agent edits go through ``LocalCalendarProvider.update_event`` and are
**not** routed through here; they keep their conservative rejection
behaviour. Granting reverse-edit to AI requires a separate, explicit
gate that is intentionally not exposed today.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..providers.calendar.factory import CalendarProviderFactory
from ..providers.calendar.resolver import CalendarAccountResolver
from .user_settings import is_two_way_sync_enabled

logger = logging.getLogger(__name__)


# Map source -> provider-name expected by CalendarProviderFactory.
_SOURCE_TO_PROVIDER = {
    "google": "google",
}

# Map source -> credential-store service identifier.
_SOURCE_TO_SERVICE = {
    "google": "google_calendar",
}


# ── Errors ──────────────────────────────────────────────────────────────


class WritebackError(Exception):
    """Base class for all writeback failures."""


class ToggleOff(WritebackError):
    """User has not enabled the two-way calendar sync setting."""


class UnsupportedSource(WritebackError):
    """The row's ``source`` has no reverse-write path."""


class SourceWriteFailed(WritebackError):
    """The provider rejected the write (e.g. Google 403, network error)."""


class CredentialsMissing(WritebackError):
    """No credentials available for the row's account."""


# ── Helpers ─────────────────────────────────────────────────────────────


def _row_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("metadata")
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw) or {}
        except (TypeError, ValueError):
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _provider_event_id(row: Dict[str, Any], meta: Dict[str, Any]) -> Optional[str]:
    """Extract the source-provider's native id.

    Prefer ``metadata.provider_event_id`` (set by newer ingest paths).
    Fallback: split the namespaced ``event_id`` ``"google:{cal}:{id}"``.
    Some Google calendar ids contain ``@`` but no ``:``; the provider id
    is always the substring after the last colon.
    """
    explicit = meta.get("provider_event_id")
    if explicit:
        return str(explicit)
    event_id = row.get("event_id") or ""
    if not event_id:
        return None
    # "google:<calendar_id>:<provider_event_id>"
    parts = event_id.split(":", 2)
    if len(parts) == 3:
        return parts[2]
    if len(parts) == 2:
        return parts[1]
    return event_id


def _calendar_id(row: Dict[str, Any], meta: Dict[str, Any]) -> Optional[str]:
    if meta.get("calendar_id"):
        return str(meta["calendar_id"])
    event_id = row.get("event_id") or ""
    parts = event_id.split(":", 2)
    if len(parts) == 3:
        return parts[1]
    return None


async def _resolve_provider(tenant_id: str, source: str, account_name: Optional[str]):
    """Build a configured calendar provider instance for this account."""
    creds = await CalendarAccountResolver.resolve_account_for_provider(
        tenant_id, source, account_name
    )
    if not creds:
        raise CredentialsMissing(
            f"No credentials for source={source!r} account={account_name!r}"
        )
    creds = dict(creds)
    creds.setdefault("provider", _SOURCE_TO_PROVIDER.get(source, source))
    provider = CalendarProviderFactory.create_provider(creds)
    if provider is None:
        raise UnsupportedSource(f"No factory for source={source!r}")
    return provider


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        from datetime import timezone

        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


# ── Public API ──────────────────────────────────────────────────────────


async def propagate_update(
    *,
    db,
    calendar_sync,
    tenant_id: str,
    existing_row: Dict[str, Any],
    fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Push an iOS edit on a non-local row to its source provider.

    ``calendar_sync`` is the ``CalendarSyncService`` instance from the
    koa app — used solely as a holder for ``_upsert_event`` so the
    refreshed cache row goes through the same code path the daily poll
    uses.

    Returns the refreshed local row (post-upsert).

    Raises ``WritebackError`` subclasses on rejection or failure.
    """
    source = (existing_row.get("source") or "").lower()
    if source == "local":
        raise UnsupportedSource("propagate_update called with local row")

    if not await is_two_way_sync_enabled(db, tenant_id):
        raise ToggleOff(
            "Two-way calendar sync is disabled. Enable it in Settings to "
            "edit events from this calendar."
        )

    meta = _row_metadata(existing_row)

    if source == "eventkit":
        return await _eventkit_local_patch(db, tenant_id, existing_row, fields)

    if source != "google":
        raise UnsupportedSource(f"Reverse edit not supported for source={source!r}")

    provider_event_id = _provider_event_id(existing_row, meta)
    calendar_id = _calendar_id(existing_row, meta) or "primary"
    account_name = meta.get("account_name") or "primary"

    if not provider_event_id:
        raise SourceWriteFailed("Could not derive provider event id from row")

    provider = await _resolve_provider(tenant_id, "google", account_name)

    update_kwargs: Dict[str, Any] = {"calendar_id": calendar_id}
    if "title" in fields:
        update_kwargs["summary"] = fields["title"]
    if "description" in fields:
        update_kwargs["description"] = fields["description"]
    if "location" in fields:
        update_kwargs["location"] = fields["location"]
    if "start_at" in fields:
        update_kwargs["start"] = _parse_ts(fields["start_at"])
    if "end_at" in fields:
        update_kwargs["end"] = _parse_ts(fields["end_at"])
    if "attendees" in fields and isinstance(fields["attendees"], list):
        emails: List[str] = []
        for a in fields["attendees"]:
            if isinstance(a, str):
                emails.append(a)
            elif isinstance(a, dict) and a.get("email"):
                emails.append(str(a["email"]))
        update_kwargs["attendees"] = emails
    if "all_day" in fields:
        update_kwargs["all_day"] = bool(fields["all_day"])
    elif existing_row.get("all_day") is not None:
        update_kwargs["all_day"] = bool(existing_row["all_day"])

    write = await provider.update_event(provider_event_id, **update_kwargs)
    if not write.get("success"):
        raise SourceWriteFailed(
            write.get("error") or "Provider rejected the update"
        )

    # Re-fetch authoritative state from Google and re-upsert.
    fetched = await provider.get_event(provider_event_id, calendar_id=calendar_id)
    if not fetched.get("success"):
        raise SourceWriteFailed(
            f"Update accepted but follow-up read failed: {fetched.get('error')}"
        )

    await calendar_sync._upsert_event(
        user_id=tenant_id,
        source="google",
        account_name=account_name,
        raw_event=fetched["data"],
        calendar_id=calendar_id,
    )

    refreshed = await db.fetchrow(
        "SELECT user_id, event_id, calendar_name, title, starts_at, ends_at, "
        "all_day, location, notes, attendees, metadata, "
        "color, recurrence_rule, reminder_minutes, "
        "source, updated_at "
        "FROM tenant_default.local_calendar_events "
        "WHERE user_id = $1 AND event_id = $2",
        tenant_id,
        existing_row["event_id"],
    )
    return dict(refreshed) if refreshed else dict(existing_row)


async def propagate_delete(
    *,
    db,
    tenant_id: str,
    existing_row: Dict[str, Any],
) -> None:
    """Push an iOS delete on a non-local row to its source provider.

    On success the local cache row is hard-deleted. Raises ``WritebackError``
    subclasses on rejection or failure.
    """
    source = (existing_row.get("source") or "").lower()
    if source == "local":
        raise UnsupportedSource("propagate_delete called with local row")

    if not await is_two_way_sync_enabled(db, tenant_id):
        raise ToggleOff(
            "Two-way calendar sync is disabled. Enable it in Settings to "
            "delete events from this calendar."
        )

    meta = _row_metadata(existing_row)

    if source == "eventkit":
        # Client owns the device-side delete. Drop the cache row so the
        # event vanishes from the merged view immediately.
        await db.execute(
            "DELETE FROM tenant_default.local_calendar_events "
            "WHERE user_id = $1 AND event_id = $2",
            tenant_id,
            existing_row["event_id"],
        )
        return

    if source != "google":
        raise UnsupportedSource(f"Reverse delete not supported for source={source!r}")

    provider_event_id = _provider_event_id(existing_row, meta)
    calendar_id = _calendar_id(existing_row, meta) or "primary"
    account_name = meta.get("account_name") or "primary"

    if not provider_event_id:
        raise SourceWriteFailed("Could not derive provider event id from row")

    provider = await _resolve_provider(tenant_id, "google", account_name)
    result = await provider.delete_event(provider_event_id, calendar_id=calendar_id)
    if not result.get("success"):
        raise SourceWriteFailed(
            result.get("error") or "Provider rejected the delete"
        )

    await db.execute(
        "DELETE FROM tenant_default.local_calendar_events "
        "WHERE user_id = $1 AND event_id = $2",
        tenant_id,
        existing_row["event_id"],
    )


async def _eventkit_local_patch(
    db, tenant_id: str, existing_row: Dict[str, Any], fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply field changes locally for an EventKit row.

    The iOS app is responsible for the EKEventStore.save call before it
    invokes us. We just patch the cache so the merged view is consistent
    with what the user just wrote on-device.
    """
    sets: List[str] = []
    args: List[Any] = []

    def add(col: str, val: Any, jsonb: bool = False) -> None:
        args.append(val)
        token = f"${len(args)}"
        if jsonb:
            token += "::jsonb"
        sets.append(f"{col} = {token}")

    if "title" in fields:
        add("title", fields["title"])
    if "start_at" in fields:
        add("starts_at", _parse_ts(fields["start_at"]))
    if "end_at" in fields:
        add("ends_at", _parse_ts(fields["end_at"]))
    if "all_day" in fields:
        add("all_day", bool(fields["all_day"]))
    if "description" in fields:
        add("notes", fields["description"])
    if "location" in fields:
        add("location", fields["location"])
    if "attendees" in fields and fields["attendees"] is not None:
        add("attendees", json.dumps(fields["attendees"]), jsonb=True)
    if "color" in fields:
        add("color", fields["color"])
    if "recurrence_rule" in fields:
        add("recurrence_rule", fields["recurrence_rule"])
    if "reminder_minutes" in fields:
        add("reminder_minutes", fields["reminder_minutes"])

    if not sets:
        return existing_row

    sets.append("updated_at = NOW()")
    args.append(tenant_id)
    args.append(existing_row["event_id"])
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
    return dict(row) if row else dict(existing_row)


__all__ = [
    "WritebackError",
    "ToggleOff",
    "UnsupportedSource",
    "SourceWriteFailed",
    "CredentialsMissing",
    "propagate_update",
    "propagate_delete",
]
