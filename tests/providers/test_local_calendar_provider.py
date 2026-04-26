"""Tests for the DB-backed LocalCalendarProvider."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from koa.providers.calendar.local import LocalCalendarProvider


def _make_db():
    db = AsyncMock()
    return db


def _row(**overrides):
    base = {
        "user_id": "user-1",
        "event_id": "evt-1",
        "calendar_name": "Default",
        "title": "Team sync",
        "starts_at": datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
        "ends_at": datetime(2026, 4, 12, 16, 0, tzinfo=timezone.utc),
        "all_day": False,
        "location": "Room A",
        "notes": "Weekly team sync",
        "attendees": None,
        "metadata": None,
        "source": "local",
        "updated_at": datetime(2026, 4, 1, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


class TestLocalCalendarProviderRead:
    @pytest.mark.asyncio
    async def test_list_events_queries_db_and_maps_rows(self):
        db = _make_db()
        db.fetch.return_value = [_row()]
        provider = LocalCalendarProvider(tenant_id="user-1", db=db)

        result = await provider.list_events(
            time_min=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 4, 13, 0, 0, tzinfo=timezone.utc),
            query="sync",
            max_results=5,
        )

        assert result["success"] is True
        assert result["count"] == 1
        event = result["data"][0]
        assert event["id"] == "evt-1"
        assert event["summary"] == "Team sync"
        assert event["description"] == "Weekly team sync"
        assert event["start"] == datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc)
        assert event["location"] == "Room A"
        assert event["source"] == "local"

        db.fetch.assert_awaited_once()
        sql_arg = db.fetch.call_args.args[0]
        assert "tenant_default.local_calendar_events" in sql_arg
        assert "ILIKE" in sql_arg

    @pytest.mark.asyncio
    async def test_list_events_returns_failure_when_db_raises(self):
        db = _make_db()
        db.fetch.side_effect = RuntimeError("db down")
        provider = LocalCalendarProvider(tenant_id="user-1", db=db)

        result = await provider.list_events()

        assert result["success"] is False
        assert "db down" in result["error"]


class TestLocalCalendarProviderWrite:
    @pytest.mark.asyncio
    async def test_create_event_inserts_local_source_with_uuid_id(self):
        db = _make_db()
        db.fetchrow.return_value = _row(event_id="local:abc-123")
        provider = LocalCalendarProvider(tenant_id="user-1", db=db)

        result = await provider.create_event(
            summary="Team sync",
            start=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 12, 16, 0, tzinfo=timezone.utc),
            description="Weekly team sync",
            location="Room A",
        )

        assert result["success"] is True
        assert result["event_id"] == "local:abc-123"
        sql_arg = db.fetchrow.call_args.args[0]
        assert "INSERT INTO tenant_default.local_calendar_events" in sql_arg
        assert "'local'" in sql_arg
        # event_id arg should be a generated local: id
        passed_event_id = db.fetchrow.call_args.args[2]
        assert passed_event_id.startswith("local:")

    @pytest.mark.asyncio
    async def test_update_event_refuses_non_local_source(self):
        db = _make_db()
        db.fetchrow.return_value = {"source": "google"}
        provider = LocalCalendarProvider(tenant_id="user-1", db=db)

        result = await provider.update_event(event_id="google:cal:1", summary="x")

        assert result["success"] is False
        assert "google" in result["error"]
        # Should not have issued an UPDATE
        assert db.execute.await_count == 0

    @pytest.mark.asyncio
    async def test_update_event_succeeds_for_local_source(self):
        db = _make_db()
        # First fetchrow: source check; second fetchrow: UPDATE RETURNING
        db.fetchrow.side_effect = [
            {"source": "local"},
            _row(title="Weekly sync", location="Room B"),
        ]
        provider = LocalCalendarProvider(tenant_id="user-1", db=db)

        result = await provider.update_event(
            event_id="local:abc",
            summary="Weekly sync",
            location="Room B",
        )

        assert result["success"] is True
        assert result["data"]["summary"] == "Weekly sync"
        assert result["data"]["location"] == "Room B"
        # The second call should be the UPDATE
        update_sql = db.fetchrow.call_args_list[1].args[0]
        assert update_sql.startswith("UPDATE tenant_default.local_calendar_events")

    @pytest.mark.asyncio
    async def test_update_event_returns_not_found_when_missing(self):
        db = _make_db()
        db.fetchrow.return_value = None
        provider = LocalCalendarProvider(tenant_id="user-1", db=db)

        result = await provider.update_event(event_id="missing", summary="x")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_delete_event_refuses_non_local_source(self):
        db = _make_db()
        db.fetchrow.return_value = {"source": "eventkit"}
        provider = LocalCalendarProvider(tenant_id="user-1", db=db)

        result = await provider.delete_event("evt-1")

        assert result["success"] is False
        assert "eventkit" in result["error"]
        assert db.execute.await_count == 0

    @pytest.mark.asyncio
    async def test_delete_event_succeeds_for_local_source(self):
        db = _make_db()
        db.fetchrow.return_value = {"source": "local"}
        provider = LocalCalendarProvider(tenant_id="user-1", db=db)

        result = await provider.delete_event("local:abc")

        assert result == {"success": True}
        db.execute.assert_awaited_once()
        sql_arg = db.execute.call_args.args[0]
        assert "DELETE FROM tenant_default.local_calendar_events" in sql_arg


class TestLocalCalendarProviderConstruction:
    def test_construct_without_db_raises(self):
        with pytest.raises(ValueError):
            LocalCalendarProvider(tenant_id="user-1", db=None)

    @pytest.mark.asyncio
    async def test_ensure_valid_token_is_always_true(self):
        provider = LocalCalendarProvider(tenant_id="user-1", db=_make_db())
        assert await provider.ensure_valid_token() is True
