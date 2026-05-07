"""Tests for ``koa.services.calendar_writeback`` — propagate_update / delete."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koa.services import calendar_writeback as cwb
from koa.services.user_settings import KEY_TWO_WAY_CALENDAR_SYNC


def _make_db(rows_to_return: List[Any] | None = None):
    db = MagicMock()
    db.execute = AsyncMock()
    fetchrow = AsyncMock()
    if rows_to_return is not None:
        fetchrow.side_effect = rows_to_return
    db.fetchrow = fetchrow
    return db


def _google_row(**overrides) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "user_id": "tenant-1",
        "event_id": "google:primary:abc123",
        "calendar_name": "primary",
        "title": "Old title",
        "starts_at": datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
        "ends_at": datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
        "all_day": False,
        "location": None,
        "notes": None,
        "attendees": None,
        "metadata": json.dumps(
            {
                "source": "google",
                "account_name": "user@example.com",
                "calendar_id": "primary",
                "provider_event_id": "abc123",
            }
        ),
        "color": None,
        "recurrence_rule": None,
        "reminder_minutes": None,
        "source": "google",
        "updated_at": datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_propagate_update_toggle_off_raises():
    """When two_way_calendar_sync is OFF the writeback rejects."""
    db = _make_db()

    async def get_setting(_db, _user, _key, default=None):
        assert _key == KEY_TWO_WAY_CALENDAR_SYNC
        return False

    with patch("koa.services.calendar_writeback.is_two_way_sync_enabled", AsyncMock(return_value=False)):
        with pytest.raises(cwb.ToggleOff):
            await cwb.propagate_update(
                db=db,
                calendar_sync=MagicMock(),
                tenant_id="tenant-1",
                existing_row=_google_row(),
                fields={"title": "New title"},
            )


@pytest.mark.asyncio
async def test_propagate_update_google_calls_provider_and_refreshes_cache():
    """Toggle ON + google source: provider write + cache refresh via re-fetch."""
    refreshed_row = _google_row(title="Authoritative title")
    db = _make_db([refreshed_row])

    provider = MagicMock()
    provider.update_event = AsyncMock(return_value={"success": True, "event_id": "abc123"})
    provider.get_event = AsyncMock(
        return_value={
            "success": True,
            "data": {
                "event_id": "abc123",
                "summary": "Authoritative title",
                "description": "",
                "start": refreshed_row["starts_at"],
                "end": refreshed_row["ends_at"],
                "all_day": False,
                "location": "",
                "attendees": [],
            },
        }
    )

    calendar_sync = MagicMock()
    calendar_sync._upsert_event = AsyncMock(return_value=True)

    with patch(
        "koa.services.calendar_writeback.is_two_way_sync_enabled",
        AsyncMock(return_value=True),
    ), patch(
        "koa.services.calendar_writeback._resolve_provider",
        AsyncMock(return_value=provider),
    ):
        result = await cwb.propagate_update(
            db=db,
            calendar_sync=calendar_sync,
            tenant_id="tenant-1",
            existing_row=_google_row(),
            fields={"title": "New title"},
        )

    provider.update_event.assert_awaited_once()
    args, kwargs = provider.update_event.call_args
    assert args[0] == "abc123"
    assert kwargs["summary"] == "New title"
    assert kwargs["calendar_id"] == "primary"
    assert kwargs["all_day"] is False  # carried from existing_row

    provider.get_event.assert_awaited_once_with("abc123", calendar_id="primary")
    calendar_sync._upsert_event.assert_awaited_once()
    upsert_kwargs = calendar_sync._upsert_event.call_args.kwargs
    assert upsert_kwargs["source"] == "google"
    assert upsert_kwargs["calendar_id"] == "primary"
    assert upsert_kwargs["raw_event"]["summary"] == "Authoritative title"

    assert result["title"] == "Authoritative title"


@pytest.mark.asyncio
async def test_propagate_update_google_all_day_flag_forwarded():
    """``all_day=True`` in fields is forwarded to provider so date-only shape is used."""
    refreshed_row = _google_row(all_day=True)
    db = _make_db([refreshed_row])

    provider = MagicMock()
    provider.update_event = AsyncMock(return_value={"success": True})
    provider.get_event = AsyncMock(
        return_value={
            "success": True,
            "data": {
                "event_id": "abc123",
                "summary": "x",
                "description": "",
                "start": datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2025, 1, 3, 0, 0, tzinfo=timezone.utc),
                "all_day": True,
                "location": "",
                "attendees": [],
            },
        }
    )

    calendar_sync = MagicMock()
    calendar_sync._upsert_event = AsyncMock()

    with patch(
        "koa.services.calendar_writeback.is_two_way_sync_enabled",
        AsyncMock(return_value=True),
    ), patch(
        "koa.services.calendar_writeback._resolve_provider",
        AsyncMock(return_value=provider),
    ):
        await cwb.propagate_update(
            db=db,
            calendar_sync=calendar_sync,
            tenant_id="tenant-1",
            existing_row=_google_row(),
            fields={
                "all_day": True,
                "start_at": "2025-01-02T00:00:00Z",
                "end_at": "2025-01-03T00:00:00Z",
            },
        )

    kwargs = provider.update_event.call_args.kwargs
    assert kwargs["all_day"] is True
    assert kwargs["start"] == datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)
    assert kwargs["end"] == datetime(2025, 1, 3, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_propagate_update_provider_failure_raises():
    """Provider returning {success: False} → SourceWriteFailed."""
    db = _make_db()
    provider = MagicMock()
    provider.update_event = AsyncMock(
        return_value={"success": False, "error": "Forbidden"}
    )

    with patch(
        "koa.services.calendar_writeback.is_two_way_sync_enabled",
        AsyncMock(return_value=True),
    ), patch(
        "koa.services.calendar_writeback._resolve_provider",
        AsyncMock(return_value=provider),
    ):
        with pytest.raises(cwb.SourceWriteFailed):
            await cwb.propagate_update(
                db=db,
                calendar_sync=MagicMock(),
                tenant_id="tenant-1",
                existing_row=_google_row(),
                fields={"title": "New"},
            )


@pytest.mark.asyncio
async def test_propagate_delete_google_hard_deletes_local_row():
    """Toggle ON + google delete → provider delete called + local row removed."""
    db = _make_db()
    provider = MagicMock()
    provider.delete_event = AsyncMock(return_value={"success": True})

    with patch(
        "koa.services.calendar_writeback.is_two_way_sync_enabled",
        AsyncMock(return_value=True),
    ), patch(
        "koa.services.calendar_writeback._resolve_provider",
        AsyncMock(return_value=provider),
    ):
        await cwb.propagate_delete(
            db=db,
            tenant_id="tenant-1",
            existing_row=_google_row(),
        )

    provider.delete_event.assert_awaited_once_with("abc123", calendar_id="primary")
    db.execute.assert_awaited_once()
    sql = db.execute.call_args.args[0]
    assert "DELETE FROM tenant_default.local_calendar_events" in sql


@pytest.mark.asyncio
async def test_propagate_update_eventkit_patches_local_only():
    """For ``source=eventkit``, no provider call — just patch the local cache."""
    eventkit_row = _google_row(
        source="eventkit",
        event_id="eventkit:EK-XYZ",
        metadata=json.dumps(
            {"source": "eventkit", "provider_event_id": "EK-XYZ"}
        ),
    )
    refreshed = dict(eventkit_row)
    refreshed["title"] = "New"
    db = _make_db([refreshed])

    with patch(
        "koa.services.calendar_writeback.is_two_way_sync_enabled",
        AsyncMock(return_value=True),
    ), patch(
        "koa.services.calendar_writeback._resolve_provider",
        AsyncMock(side_effect=AssertionError("must not resolve provider for eventkit")),
    ):
        result = await cwb.propagate_update(
            db=db,
            calendar_sync=MagicMock(),
            tenant_id="tenant-1",
            existing_row=eventkit_row,
            fields={"title": "New"},
        )

    db.fetchrow.assert_awaited_once()
    sql = db.fetchrow.call_args.args[0]
    assert "UPDATE tenant_default.local_calendar_events" in sql
    assert result["title"] == "New"


@pytest.mark.asyncio
async def test_propagate_delete_eventkit_drops_local_row():
    eventkit_row = _google_row(
        source="eventkit",
        event_id="eventkit:EK-XYZ",
    )
    db = _make_db()

    with patch(
        "koa.services.calendar_writeback.is_two_way_sync_enabled",
        AsyncMock(return_value=True),
    ):
        await cwb.propagate_delete(
            db=db,
            tenant_id="tenant-1",
            existing_row=eventkit_row,
        )

    db.execute.assert_awaited_once()
    sql = db.execute.call_args.args[0]
    assert "DELETE FROM tenant_default.local_calendar_events" in sql
