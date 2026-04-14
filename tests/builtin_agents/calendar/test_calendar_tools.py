from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from koa.builtin_agents.calendar.agent import CalendarAgent
from koa.builtin_agents.calendar.tools import (
    _preview_delete_event,
    _resolve_calendar_provider,
    check_upcoming_events,
    create_event,
    delete_event,
    query_events,
    update_event,
)
from koa.builtin_agents.shared.routing_preferences import (
    ResolvedSurfaceTarget,
    set_routing_preference,
)
from koa.models import AgentToolContext, ToolOutput


class DummyCalendarProvider:
    async def ensure_valid_token(self):
        return True

    async def list_events(self, **kwargs):
        return {
            "success": True,
            "data": [
                {
                    "id": "evt-1",
                    "event_id": "evt-1",
                    "summary": "Team sync",
                    "description": "Weekly team sync",
                    "start": datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 4, 12, 16, 0, tzinfo=timezone.utc),
                    "location": "Room A",
                }
            ],
        }

    async def create_event(self, **kwargs):
        return {"success": True, "event_id": "evt-1"}

    async def update_event(self, **kwargs):
        return {"success": True, "event_id": kwargs["event_id"]}

    async def delete_event(self, event_id, **kwargs):
        return {"success": True, "event_id": event_id}


class FailingCalendarProvider:
    async def list_events(self, **kwargs):
        return {"success": False, "error": "backend down"}


class DeleteFailingCalendarProvider(DummyCalendarProvider):
    async def delete_event(self, event_id, **kwargs):
        return {"success": False, "error": "backend down"}


def _context() -> AgentToolContext:
    return AgentToolContext(
        tenant_id="user-1",
        metadata={"timezone": "UTC", "koiai_url": "https://koiai.example", "service_key": "svc-key"},
    )


class TestCalendarToolSchema:
    def test_calendar_tools_accept_explicit_target_arguments(self):
        for tool in (query_events, create_event, update_event, delete_event):
            assert "target_provider" in tool.parameters["properties"]
            assert "target_account" in tool.parameters["properties"]


class TestCalendarToolRouting:
    @pytest.mark.asyncio
    async def test_query_events_uses_resolved_provider(self):
        provider = DummyCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await query_events.executor(
                {
                    "time_range": "today",
                    "target_provider": "local",
                },
                _context(),
            )

        assert isinstance(result, ToolOutput)
        assert "Found 1 event(s) today:" in result.text
        assert '"eventId": "evt-1"' in result.media[0]["data"]

    @pytest.mark.asyncio
    async def test_create_event_uses_resolved_provider(self):
        provider = DummyCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await create_event.executor(
                {
                    "summary": "Team sync",
                    "start": "2026-04-12 15:00",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "added 'Team sync' to your calendar" in result

    @pytest.mark.asyncio
    async def test_update_event_uses_resolved_provider(self):
        provider = DummyCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await update_event.executor(
                {
                    "target": "team sync",
                    "changes": {"new_title": "Weekly sync"},
                    "target_provider": "local",
                },
                _context(),
            )

        assert 'renamed the event to "Weekly sync"' in result

    @pytest.mark.asyncio
    async def test_delete_event_uses_resolved_provider(self):
        provider = DummyCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.search_helper.search_calendar_events",
            side_effect=AssertionError("search_calendar_events should not be used"),
        ), patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await delete_event.executor(
                {
                    "search_query": "team sync",
                    "time_range": "today",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "removed 1 event(s)" in result

    @pytest.mark.asyncio
    async def test_preview_delete_event_uses_resolved_provider(self):
        provider = DummyCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.search_helper.search_calendar_events",
            side_effect=AssertionError("search_calendar_events should not be used"),
        ), patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await _preview_delete_event(
                {
                    "search_query": "team sync",
                    "time_range": "today",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "Found 1 event:" in result
        assert "Delete it?" in result

    @pytest.mark.asyncio
    async def test_resolve_calendar_provider_uses_provider_specific_account_resolution(self):
        provider = DummyCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools.LocalBackendClient.from_context",
            return_value=object(),
            create=True,
        ), patch(
            "koa.builtin_agents.calendar.tools.resolve_surface_target",
            new=AsyncMock(
                return_value=ResolvedSurfaceTarget(
                    surface="calendar",
                    provider="google",
                    account="work",
                    source="saved",
                )
            ),
        ), patch(
            "koa.providers.calendar.factory.CalendarProviderFactory.get_supported_providers",
            return_value=["google"],
        ), patch(
            "koa.providers.calendar.resolver.CalendarAccountResolver.resolve_account_for_provider",
            new=AsyncMock(return_value={"provider": "google", "account_name": "work"}),
        ), patch(
            "koa.providers.calendar.resolver.CalendarAccountResolver.resolve_account",
            new=AsyncMock(side_effect=AssertionError("generic resolve_account should not be used")),
            create=True,
        ), patch(
            "koa.providers.calendar.factory.CalendarProviderFactory.create_provider",
            return_value=provider,
        ):
            resolved_provider, account, error = await _resolve_calendar_provider(_context())

        assert error is None
        assert resolved_provider is provider
        assert account["provider"] == "google"
        assert account["account_name"] == "work"

    @pytest.mark.asyncio
    async def test_delete_event_returns_wrapped_error_when_resolved_provider_fails(self):
        provider = FailingCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await delete_event.executor(
                {
                    "search_query": "team sync",
                    "time_range": "today",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "couldn't finish that calendar action" in result.lower()
        assert "save it locally" in result.lower()

    @pytest.mark.asyncio
    async def test_preview_delete_event_returns_wrapped_error_when_resolved_provider_fails(self):
        provider = FailingCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await _preview_delete_event(
                {
                    "search_query": "team sync",
                    "time_range": "today",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "couldn't retrieve your calendar data" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_query_events_wraps_preference_lookup_failures(self):
        backend_client = AsyncMock()
        backend_client.get_routing_preference.side_effect = RuntimeError("backend down")

        with patch(
            "koa.builtin_agents.calendar.tools.LocalBackendClient.from_context",
            return_value=backend_client,
        ):
            result = await query_events.executor(
                {
                    "time_range": "today",
                },
                _context(),
            )

        assert "couldn't retrieve your calendar data" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_preview_delete_event_wraps_preference_lookup_failures(self):
        backend_client = AsyncMock()
        backend_client.get_routing_preference.side_effect = RuntimeError("backend down")

        with patch(
            "koa.builtin_agents.calendar.tools.LocalBackendClient.from_context",
            return_value=backend_client,
        ):
            result = await _preview_delete_event(
                {
                    "search_query": "team sync",
                    "time_range": "today",
                },
                _context(),
            )

        assert "couldn't retrieve your calendar data" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_check_upcoming_events_wraps_preference_lookup_failures(self):
        backend_client = AsyncMock()
        backend_client.get_routing_preference.side_effect = RuntimeError("backend down")

        with patch(
            "koa.builtin_agents.calendar.tools.LocalBackendClient.from_context",
            return_value=backend_client,
        ):
            result = await check_upcoming_events.executor({}, _context())

        assert "couldn't retrieve your calendar data" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_create_event_wraps_preference_lookup_failures(self):
        backend_client = AsyncMock()
        backend_client.get_routing_preference.side_effect = RuntimeError("backend down")

        with patch(
            "koa.builtin_agents.calendar.tools.LocalBackendClient.from_context",
            return_value=backend_client,
        ):
            result = await create_event.executor(
                {
                    "summary": "Team lunch",
                    "start": "2026-04-12T12:00:00",
                    "end": "2026-04-12T13:00:00",
                },
                _context(),
            )

        assert "couldn't finish that calendar action" in result.lower()
        assert "save it locally" in result.lower()

    @pytest.mark.asyncio
    async def test_update_event_wraps_preference_lookup_failures(self):
        backend_client = AsyncMock()
        backend_client.get_routing_preference.side_effect = RuntimeError("backend down")

        with patch(
            "koa.builtin_agents.calendar.tools.LocalBackendClient.from_context",
            return_value=backend_client,
        ):
            result = await update_event.executor(
                {
                    "target": "team sync",
                    "changes": {"new_title": "Weekly sync"},
                },
                _context(),
            )

        assert "couldn't finish that calendar action" in result.lower()
        assert "save it locally" in result.lower()

    @pytest.mark.asyncio
    async def test_update_event_returns_wrapped_error_when_search_fails(self):
        provider = FailingCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await update_event.executor(
                {
                    "target": "team sync",
                    "changes": {"new_title": "Weekly sync"},
                    "target_provider": "local",
                },
                _context(),
            )

        assert "couldn't finish that calendar action" in result.lower()
        assert "save it locally" in result.lower()

    @pytest.mark.asyncio
    async def test_delete_event_returns_wrapped_error_when_all_deletes_fail(self):
        provider = DeleteFailingCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await delete_event.executor(
                {
                    "search_query": "team sync",
                    "time_range": "today",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "couldn't finish that calendar action" in result.lower()
        assert "save it locally" in result.lower()

    @pytest.mark.asyncio
    async def test_query_events_read_failure_does_not_suggest_save_locally(self):
        provider = FailingCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await query_events.executor(
                {
                    "time_range": "today",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "save it locally" not in result.lower()
        assert "couldn't retrieve your calendar data" in result.lower()

    @pytest.mark.asyncio
    async def test_check_upcoming_events_read_failure_does_not_suggest_save_locally(self):
        provider = FailingCalendarProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await check_upcoming_events.executor({}, _context())

        assert "save it locally" not in result.lower()
        assert "couldn't retrieve your calendar data" in result.lower()


class TestCalendarAgent:
    def test_calendar_agent_allows_local_routing_and_preference_changes(self):
        assert "requires_service" not in CalendarAgent._valet_metadata.extra
        assert set_routing_preference in CalendarAgent.tools
        assert "set_routing_preference" in CalendarAgent._SYSTEM_PROMPT_TEMPLATE
        assert "default destination" in CalendarAgent._SYSTEM_PROMPT_TEMPLATE
