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
        metadata={
            "timezone": "UTC",
            "koiai_url": "https://koiai.example",
            "service_key": "svc-key",
        },
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
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
                    "target_provider": "local",
                },
                _context(),
            )

        assert isinstance(result, ToolOutput)
        assert "Found 1 event(s) between" in result.text
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
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
                    "target_provider": "local",
                },
                _context(),
            )

        assert 'renamed the event to "Weekly sync"' in result

    @pytest.mark.asyncio
    async def test_delete_event_uses_resolved_provider(self):
        provider = DummyCalendarProvider()

        with (
            patch(
                "koa.builtin_agents.calendar.search_helper.search_calendar_events",
                side_effect=AssertionError("search_calendar_events should not be used"),
            ),
            patch(
                "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
                new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
                create=True,
            ),
        ):
            result = await delete_event.executor(
                {
                    "search_query": "team sync",
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "removed 1 event(s)" in result

    @pytest.mark.asyncio
    async def test_preview_delete_event_uses_resolved_provider(self):
        provider = DummyCalendarProvider()

        with (
            patch(
                "koa.builtin_agents.calendar.search_helper.search_calendar_events",
                side_effect=AssertionError("search_calendar_events should not be used"),
            ),
            patch(
                "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
                new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
                create=True,
            ),
        ):
            result = await _preview_delete_event(
                {
                    "search_query": "team sync",
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "Found 1 event:" in result
        assert "Delete it?" in result

    @pytest.mark.asyncio
    async def test_resolve_calendar_provider_uses_provider_specific_account_resolution(self):
        provider = DummyCalendarProvider()

        with (
            patch(
                "koa.builtin_agents.calendar.tools.resolve_surface_target",
                new=AsyncMock(
                    return_value=ResolvedSurfaceTarget(
                        surface="calendar",
                        provider="google",
                        account="work",
                        source="saved",
                    )
                ),
            ),
            patch(
                "koa.providers.calendar.factory.CalendarProviderFactory.get_supported_providers",
                return_value=["google"],
            ),
            patch(
                "koa.providers.calendar.resolver.CalendarAccountResolver.resolve_account_for_provider",
                new=AsyncMock(return_value={"provider": "google", "account_name": "work"}),
            ),
            patch(
                "koa.providers.calendar.resolver.CalendarAccountResolver.resolve_account",
                new=AsyncMock(
                    side_effect=AssertionError("generic resolve_account should not be used")
                ),
                create=True,
            ),
            patch(
                "koa.providers.calendar.factory.CalendarProviderFactory.create_provider",
                return_value=provider,
            ),
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
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
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
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "couldn't retrieve your calendar data" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_query_events_wraps_preference_lookup_failures(self):
        with patch(
            "koa.builtin_agents.calendar.tools.resolve_surface_target",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
        ):
            result = await query_events.executor(
                {
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
                },
                _context(),
            )

        assert "couldn't retrieve your calendar data" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_preview_delete_event_wraps_preference_lookup_failures(self):
        with patch(
            "koa.builtin_agents.calendar.tools.resolve_surface_target",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
        ):
            result = await _preview_delete_event(
                {
                    "search_query": "team sync",
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
                },
                _context(),
            )

        assert "couldn't retrieve your calendar data" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_check_upcoming_events_wraps_preference_lookup_failures(self):
        with patch(
            "koa.builtin_agents.calendar.tools.resolve_surface_target",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
        ):
            result = await check_upcoming_events.executor({}, _context())

        assert "couldn't retrieve your calendar data" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_create_event_wraps_preference_lookup_failures(self):
        with patch(
            "koa.builtin_agents.calendar.tools.resolve_surface_target",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
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
        with patch(
            "koa.builtin_agents.calendar.tools.resolve_surface_target",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
        ):
            result = await update_event.executor(
                {
                    "target": "team sync",
                    "changes": {"new_title": "Weekly sync"},
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
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
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
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
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
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
                    "time_min": "2026-04-12",
                    "time_max": "2026-04-13",
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

    def test_calendar_agent_prompt_documents_update_event_time_window(self):
        prompt = CalendarAgent._SYSTEM_PROMPT_TEMPLATE
        # update_event must be in the time-window rules alongside query/delete.
        assert "update_event" in prompt
        assert "query_events, update_event, and delete_event" in prompt
        # Past windows must be allowed for updates so we can edit historical
        # events (the production bug we just fixed).
        assert "even if it's in the past" in prompt


# ---------------------------------------------------------------------------
# update_event — past-window / ambiguity / missing-bounds regressions
# ---------------------------------------------------------------------------


class _MultiMatchProvider:
    """Provider that returns several events whose titles all contain the target."""

    async def list_events(self, **kwargs):
        return {
            "success": True,
            "data": [
                {
                    "id": "evt-mon",
                    "summary": "SF trip - Monday",
                    "start": datetime(2026, 4, 27, 9, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 4, 27, 17, 0, tzinfo=timezone.utc),
                    "location": "San Francisco",
                },
                {
                    "id": "evt-tue",
                    "summary": "SF trip - Tuesday",
                    "start": datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 4, 28, 17, 0, tzinfo=timezone.utc),
                    "location": "San Francisco",
                },
                {
                    "id": "evt-wed",
                    "summary": "SF trip - Wednesday",
                    "start": datetime(2026, 4, 29, 9, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 4, 29, 17, 0, tzinfo=timezone.utc),
                    "location": "San Francisco",
                },
            ],
        }

    async def update_event(self, **kwargs):  # should never be called
        raise AssertionError("update_event must not mutate when there are multiple matches")


class _NoMatchProvider:
    async def list_events(self, **kwargs):
        return {"success": True, "data": []}

    async def update_event(self, **kwargs):  # should never be called
        raise AssertionError("update_event must not mutate when target is not found")


class _SinglePastMatchProvider:
    """Provider that returns exactly one matching past event (4/27 SF trip)."""

    def __init__(self):
        self.update_calls: list = []

    async def list_events(self, **kwargs):
        return {
            "success": True,
            "data": [
                {
                    "id": "evt-sf-27",
                    "event_id": "evt-sf-27",
                    "summary": "SF business trip",
                    "description": "OnePoint offsite",
                    "start": datetime(2026, 4, 27, 9, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 4, 27, 17, 0, tzinfo=timezone.utc),
                    "location": "San Francisco",
                }
            ],
        }

    async def update_event(self, **kwargs):
        self.update_calls.append(kwargs)
        return {"success": True, "event_id": kwargs["event_id"]}


class TestUpdateEventWindowing:
    @pytest.mark.asyncio
    async def test_update_event_finds_past_event_within_supplied_window(self):
        """Regression: previously hardcoded now→now+30d window made past events invisible."""
        provider = _SinglePastMatchProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await update_event.executor(
                {
                    "target": "SF",
                    "changes": {"new_title": "SF trip (rescheduled)"},
                    # Past window — would have been skipped by the old hardcoded future-only search.
                    "time_min": "2026-04-27",
                    "time_max": "2026-05-01",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "renamed the event" in result.lower()
        assert len(provider.update_calls) == 1
        assert provider.update_calls[0]["event_id"] == "evt-sf-27"
        assert provider.update_calls[0]["summary"] == "SF trip (rescheduled)"

    @pytest.mark.asyncio
    async def test_update_event_returns_disambiguation_when_multiple_matches(self):
        """Multiple matches must NOT silently mutate the first hit."""
        provider = _MultiMatchProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await update_event.executor(
                {
                    "target": "SF trip",
                    "changes": {"new_location": "Mountain View"},
                    "time_min": "2026-04-27",
                    "time_max": "2026-05-01",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "found 3 events matching 'SF trip'" in result
        assert "Monday" in result and "Tuesday" in result and "Wednesday" in result
        assert "which one to update" in result.lower()

    @pytest.mark.asyncio
    async def test_update_event_returns_friendly_not_found_message(self):
        provider = _NoMatchProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await update_event.executor(
                {
                    "target": "ghost meeting",
                    "changes": {"new_title": "Renamed"},
                    "time_min": "2026-04-27",
                    "time_max": "2026-05-01",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "couldn't find" in result.lower()
        # Must surface the actual window so the LLM can self-correct.
        assert "2026-04-27" in result and "2026-05-01" in result

    @pytest.mark.asyncio
    async def test_update_event_surfaces_time_bound_validation_errors(self):
        provider = _SinglePastMatchProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await update_event.executor(
                {
                    "target": "SF",
                    "changes": {"new_title": "X"},
                    "time_min": "2026-05-01",
                    "time_max": "2026-04-27",  # inverted on purpose
                    "target_provider": "local",
                },
                _context(),
            )

        assert "cannot update event" in result.lower()
        assert "time_max must be strictly after time_min" in result
        assert provider.update_calls == []

    @pytest.mark.asyncio
    async def test_update_event_schema_requires_time_window(self):
        """Tool schema must mark time_min/time_max as required so the LLM is forced to supply them."""
        required = update_event.parameters.get("required") or []
        assert "time_min" in required
        assert "time_max" in required


class TestUpdateEventPreview:
    @pytest.mark.asyncio
    async def test_preview_resolves_single_match_and_shows_window(self):
        from koa.builtin_agents.calendar.tools import _preview_update_event

        provider = _SinglePastMatchProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            preview = await _preview_update_event(
                {
                    "target": "SF",
                    "changes": {"new_title": "SF trip (rescheduled)"},
                    "time_min": "2026-04-27",
                    "time_max": "2026-05-01",
                    "target_provider": "local",
                },
                _context(),
            )

        # Window is visible to the user so they can sanity-check before approving.
        assert "2026-04-27" in preview and "2026-05-01" in preview
        assert "Resolved event:" in preview
        assert "SF business trip" in preview
        assert "Title -> SF trip (rescheduled)" in preview

    @pytest.mark.asyncio
    async def test_preview_surfaces_ambiguity_before_approval(self):
        from koa.builtin_agents.calendar.tools import _preview_update_event

        provider = _MultiMatchProvider()

        with patch(
            "koa.builtin_agents.calendar.tools._resolve_calendar_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            preview = await _preview_update_event(
                {
                    "target": "SF trip",
                    "changes": {"new_location": "Mountain View"},
                    "time_min": "2026-04-27",
                    "time_max": "2026-05-01",
                    "target_provider": "local",
                },
                _context(),
            )

        assert "found 3 events" in preview.lower()
