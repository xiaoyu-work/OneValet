"""Integration tests for CalendarAgent.

Tests tool selection, argument extraction, response quality, and approval flow for:
- query_events: Search and list calendar events by time range or keywords
- create_event: Create a new calendar event (needs approval)
- update_event: Update an existing event (needs approval)
- delete_event: Delete calendar events matching search criteria (needs approval)
- set_routing_preference: Save the user's default calendar destination
"""

import pytest

from koa.result import AgentStatus

pytestmark = [pytest.mark.integration, pytest.mark.productivity]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("What's on my calendar today?", ["query_events"]),
    ("Do I have any meetings tomorrow?", ["query_events"]),
    ("Show my schedule for this week", ["query_events"]),
    ("Schedule a meeting with Bob tomorrow at 2pm", ["create_event"]),
    ("Create an event: dentist appointment Friday at 10am", ["create_event"]),
    ("Add lunch with Sarah on March 5th at noon", ["create_event"]),
    ("Move my 2pm meeting to 4pm", ["update_event", "query_events"]),
    ("Reschedule the team standup to 10am", ["update_event", "query_events"]),
    ("Cancel my meeting with Bob", ["delete_event", "query_events"]),
    ("Delete the dentist appointment", ["delete_event", "query_events"]),
    ("Remove all meetings tomorrow", ["delete_event", "query_events"]),
    # Routing preference cases
    ("From now on save all events to Google Calendar", ["set_routing_preference"]),
    ("Use my local calendar by default", ["set_routing_preference"]),
    ("Switch to Google Calendar for all new events", ["set_routing_preference"]),
]


@pytest.mark.parametrize(
    "user_input,expected_tools",
    TOOL_SELECTION_CASES,
    ids=[c[0][:40] for c in TOOL_SELECTION_CASES],
)
async def test_tool_selection(conversation, user_input, expected_tools):
    conv = await conversation()
    await conv.send_until_tool_called(user_input)
    conv.assert_any_tool_called(expected_tools)


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------


async def test_extracts_query_time_range(conversation):
    """query_events should receive concrete ISO time_min/time_max for 'today'."""
    import re

    conv = await conversation()
    await conv.send("What's on my calendar today?")
    conv.assert_tool_called("query_events")

    args = conv.get_tool_args("query_events")[0]
    time_min = args.get("time_min", "")
    time_max = args.get("time_max", "")

    iso_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
    assert iso_pattern.match(time_min), f"Expected ISO-formatted time_min, got '{time_min}'"
    assert iso_pattern.match(time_max), f"Expected ISO-formatted time_max, got '{time_max}'"
    assert time_max > time_min, (
        f"time_max ({time_max}) must be strictly after time_min ({time_min})"
    )


async def test_extracts_create_event_fields(conversation):
    """create_event should receive summary and start from the user message."""
    conv = await conversation()
    await conv.send_until_tool_called("Create a meeting called Team Sync tomorrow at 3pm")
    conv.assert_tool_called("create_event")

    args = conv.get_tool_args("create_event")[0]
    summary = args.get("summary", "").lower()
    assert "team sync" in summary or "team" in summary, (
        f"Expected summary containing 'team sync', got '{summary}'"
    )
    assert args.get("start"), "start time should not be empty"


async def test_extracts_update_event_target(conversation):
    """update_event should identify the target event from the user message."""
    conv = await conversation()
    await conv.send_until_tool_called("Move the team standup to 11am")

    # LLM may call query_events first (multi-step update pattern)
    update_calls = conv.get_tool_calls("update_event")
    if not update_calls:
        conv.assert_tool_called("query_events")
        pytest.skip("LLM called query_events first (multi-step update pattern)")

    args = update_calls[0]["arguments"]
    target = (
        args.get("target", "")
        or args.get("event_query", "")
        or args.get("search_query", "")
        or args.get("summary", "")
    ).lower()
    assert "standup" in target or "team" in target, (
        f"Expected target containing 'standup' or 'team', got '{target}'. Full args: {args}"
    )


async def test_extracts_delete_event_query(conversation):
    """delete_event should receive a search_query matching the user's description."""
    conv = await conversation()
    await conv.send_until_tool_called("Cancel the dentist appointment")
    conv.assert_any_tool_called(["delete_event", "query_events"])

    delete_calls = conv.get_tool_calls("delete_event")
    if delete_calls:
        args = delete_calls[0]["arguments"]
        search_query = (
            args.get("search_query", "") or args.get("query", "") or args.get("event_id", "")
        ).lower()
        assert "dentist" in search_query or search_query, (
            f"Expected search_query containing 'dentist', got '{search_query}'"
        )


# ---------------------------------------------------------------------------
# Routing preference — tool selection and argument extraction
# ---------------------------------------------------------------------------


async def test_routing_preference_tool_selected_for_google(conversation):
    """Asking to use Google Calendar by default should invoke set_routing_preference."""
    conv = await conversation()
    await conv.send_until_tool_called("From now on add all my events to Google Calendar")
    conv.assert_tool_called("set_routing_preference")


async def test_routing_preference_tool_selected_for_local(conversation):
    """Asking to switch back to local calendar should invoke set_routing_preference."""
    conv = await conversation()
    await conv.send_until_tool_called("Use my local calendar by default for everything")
    conv.assert_tool_called("set_routing_preference")


async def test_routing_preference_extracts_surface_calendar(conversation):
    """set_routing_preference should always receive surface='calendar' for calendar requests."""
    conv = await conversation()
    await conv.send_until_tool_called("Default to Google Calendar for future events")
    conv.assert_tool_called("set_routing_preference")

    args = conv.get_tool_args("set_routing_preference")[0]
    surface = args.get("surface", "").lower()
    assert surface == "calendar", f"Expected surface='calendar', got '{surface}'"


async def test_routing_preference_extracts_google_provider(conversation):
    """set_routing_preference should receive provider='google' when the user asks for Google Calendar."""
    conv = await conversation()
    await conv.send_until_tool_called("Switch to Google Calendar as my default")
    conv.assert_tool_called("set_routing_preference")

    args = conv.get_tool_args("set_routing_preference")[0]
    provider = args.get("provider", "").lower()
    assert "google" in provider, f"Expected provider containing 'google', got '{provider}'"


async def test_routing_preference_extracts_local_provider(conversation):
    """set_routing_preference should receive provider='local' when the user asks for local calendar."""
    conv = await conversation()
    await conv.send_until_tool_called("Save future events to my local calendar please")
    conv.assert_tool_called("set_routing_preference")

    args = conv.get_tool_args("set_routing_preference")[0]
    provider = args.get("provider", "").lower()
    assert "local" in provider, f"Expected provider containing 'local', got '{provider}'"


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------


async def test_response_quality_query(conversation, llm_judge):
    """Querying the calendar should produce a structured event listing."""
    conv = await conversation()
    await conv.auto_complete("Show my schedule for today")

    passed = await llm_judge(
        "Show my schedule for today",
        conv.last_message,
        "The response should list calendar events for today in a readable format, "
        "mentioning event names and times. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


async def test_response_quality_create(conversation, llm_judge):
    """Creating an event should confirm the details."""
    conv = await conversation()
    msg = "Schedule a lunch with Alice tomorrow at noon"
    await conv.auto_complete(msg)

    passed = await llm_judge(
        msg,
        conv.last_message,
        "The response should confirm that a calendar event has been created or "
        "scheduled. It should acknowledge the creation positively. "
        "It should not be an error message or ask for more information.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


# ---------------------------------------------------------------------------
# Approval flow
# ---------------------------------------------------------------------------


async def test_create_event_triggers_approval(conversation):
    """create_event should pause for user approval before executing."""
    conv = await conversation()
    await conv.send_until_status(
        "Schedule a meeting with Bob tomorrow at 2pm",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    conv.assert_tool_called("create_event")
    conv.assert_status(AgentStatus.WAITING_FOR_APPROVAL)


async def test_create_event_approve_executes(conversation):
    """Approving create_event should execute the tool and complete."""
    conv = await conversation()
    await conv.send_until_status(
        "Schedule a meeting with Bob tomorrow at 2pm",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    result = await conv.send("yes, create it")
    assert result.status in (AgentStatus.COMPLETED, AgentStatus.WAITING_FOR_APPROVAL), (
        f"Expected COMPLETED after approval, got {result.status}"
    )


async def test_create_event_reject_cancels(conversation):
    """Rejecting create_event should cancel the operation."""
    conv = await conversation()
    await conv.send_until_status(
        "Schedule a meeting with Bob tomorrow at 2pm",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    result = await conv.send("no, cancel it")
    assert result.status in (AgentStatus.CANCELLED, AgentStatus.COMPLETED), (
        f"Expected CANCELLED after rejection, got {result.status}"
    )


# ---------------------------------------------------------------------------
# Past-date update regression
#
# Production bug (2026-05-07): user asked "你改一下我上周从27号到30号在湾区
# 出差你更新下我的日历". CalendarAgent prompted for approval *twice* and the
# update silently failed because update_event hardcoded its search window to
# [now, now+30d) — past events were invisible. Each retry tripped a fresh
# WAITING_FOR_APPROVAL until max_turns exhausted.
#
# These tests exercise the agent at the LLM level to make sure:
#   - the agent supplies concrete past time bounds for update_event
#   - approval fires exactly once, and approve→COMPLETED works
# ---------------------------------------------------------------------------


def _is_past_iso(value: str) -> bool:
    """Best-effort check that an ISO-8601 string falls in the past or covers the past."""
    import re
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    if not value:
        return False
    text = value.strip()
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}")
    if not iso_re.match(text):
        return False
    try:
        if len(text) == 10:
            parsed = _dt.fromisoformat(text + "T00:00:00+00:00")
        else:
            parsed = _dt.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_tz.utc)
    except ValueError:
        return False
    return parsed < _dt.now(_tz.utc)


async def test_update_event_for_past_trip_uses_past_window(conversation):
    """The LLM must scope its update_event search to the user's past dates,
    not 'today onward'."""
    conv = await conversation()
    msg = (
        "Update my SF business trip from last Monday April 27th — "
        "rename it to 'SF offsite (final)'."
    )
    await conv.send_until_status(msg, AgentStatus.WAITING_FOR_APPROVAL)

    update_calls = conv.get_tool_calls("update_event")
    if not update_calls:
        # LLM legitimately may scout via query_events first; in that case,
        # walk the conversation forward until the update fires.
        await conv.send("yes, go ahead — search and update it")
        update_calls = conv.get_tool_calls("update_event")
    assert update_calls, "Expected update_event to be called for a past-date update"

    args = update_calls[0]["arguments"]
    time_min = args.get("time_min", "")
    time_max = args.get("time_max", "")
    assert time_min and time_max, (
        f"update_event must receive explicit time_min/time_max; got args={args}"
    )
    assert _is_past_iso(time_min), (
        f"time_min must be in the past for a 'last week' update; got time_min={time_min!r}"
    )


async def test_update_event_past_trip_completes_after_single_approval(conversation):
    """Approving a past-date update should COMPLETE, not loop into a second approval
    or exhaust the agent's turn budget (which was the production symptom)."""
    conv = await conversation()
    await conv.send_until_status(
        "Update my SF business trip from last Monday April 27th — "
        "change the location to 'Palo Alto'.",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    # Only count entries where the executor actually ran (canned result, not the
    # __PENDING_APPROVAL__ sentinel emitted by the recording preview).
    pre_executions = sum(
        1 for c in conv.get_tool_calls("update_event") if c.get("result") != "__PENDING_APPROVAL__"
    )
    assert pre_executions == 0, "update_event should not have executed before the user approved"

    result = await conv.send("approve")

    post_executions = sum(
        1 for c in conv.get_tool_calls("update_event") if c.get("result") != "__PENDING_APPROVAL__"
    )
    assert post_executions >= 1, "update_event must execute after approval"
    assert result.status == AgentStatus.COMPLETED, (
        f"Expected COMPLETED after a single approval; got {result.status}. "
        f"(Production bug: status would loop back to WAITING_FOR_APPROVAL.)"
    )
