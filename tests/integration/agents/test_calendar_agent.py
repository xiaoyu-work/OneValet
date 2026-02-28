"""Integration tests for CalendarAgent.

Tests tool selection, argument extraction, and response quality for:
- query_events: Search and list calendar events by time range or keywords
- create_event: Create a new calendar event
- update_event: Update an existing event (reschedule, rename, change location)
- delete_event: Delete calendar events matching search criteria
"""

import pytest

pytestmark = [pytest.mark.integration]


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
    ("Move my 2pm meeting to 4pm", ["update_event"]),
    ("Reschedule the team standup to 10am", ["update_event"]),
    ("Cancel my meeting with Bob", ["delete_event"]),
    ("Delete the dentist appointment", ["delete_event"]),
    ("Remove all meetings tomorrow", ["delete_event"]),
]


@pytest.mark.parametrize(
    "user_input,expected_tools",
    TOOL_SELECTION_CASES,
    ids=[c[0][:40] for c in TOOL_SELECTION_CASES],
)
async def test_tool_selection(orchestrator_factory, user_input, expected_tools):
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", user_input)
    tools_called = [c["tool_name"] for c in recorder.tool_calls]
    assert any(t in tools_called for t in expected_tools), (
        f"Expected one of {expected_tools}, got {tools_called}"
    )


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

async def test_extracts_query_time_range(orchestrator_factory):
    """query_events should receive an appropriate time_range for 'today'."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "What's on my calendar today?")

    query_calls = [c for c in recorder.tool_calls if c["tool_name"] == "query_events"]
    assert query_calls, "query_events was never called"

    args = query_calls[0]["arguments"]
    time_range = args.get("time_range", "").lower()
    assert "today" in time_range, (
        f"Expected time_range containing 'today', got '{time_range}'"
    )


async def test_extracts_create_event_fields(orchestrator_factory):
    """create_event should receive summary and start from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user", "Create a meeting called Team Sync tomorrow at 3pm"
    )

    create_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "create_event"
    ]
    assert create_calls, "create_event was never called"

    args = create_calls[0]["arguments"]
    summary = args.get("summary", "").lower()
    assert "team sync" in summary or "team" in summary, (
        f"Expected summary containing 'team sync', got '{summary}'"
    )
    assert args.get("start"), "start time should not be empty"


async def test_extracts_update_event_target_and_changes(orchestrator_factory):
    """update_event should identify the target event and the requested changes."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user", "Move the team standup to 11am"
    )

    update_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "update_event"
    ]
    assert update_calls, "update_event was never called"

    args = update_calls[0]["arguments"]
    target = args.get("target", "").lower()
    assert "standup" in target or "team" in target, (
        f"Expected target containing 'standup' or 'team', got '{target}'"
    )
    changes = args.get("changes", {})
    assert changes, "changes dict should not be empty"
    # The new time should appear in new_time
    new_time = changes.get("new_time", "")
    assert new_time, "new_time should be set in changes"


async def test_extracts_delete_event_query(orchestrator_factory):
    """delete_event should receive a search_query matching the user's description."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Cancel the dentist appointment")

    delete_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "delete_event"
    ]
    assert delete_calls, "delete_event was never called"

    args = delete_calls[0]["arguments"]
    search_query = args.get("search_query", "").lower()
    assert "dentist" in search_query, (
        f"Expected search_query containing 'dentist', got '{search_query}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_query(orchestrator_factory, llm_judge):
    """Querying the calendar should produce a structured event listing."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Show my schedule for today")

    passed = await llm_judge(
        "Show my schedule for today",
        result,
        "The response should list calendar events for today in a readable format, "
        "mentioning event names and times. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result}"


async def test_response_quality_create(orchestrator_factory, llm_judge):
    """Creating an event should confirm the details."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Schedule a lunch with Alice tomorrow at noon"
    )

    passed = await llm_judge(
        "Schedule a lunch with Alice tomorrow at noon",
        result,
        "The response should confirm that a calendar event has been created. "
        "It should mention the event name (lunch with Alice) and the time (noon).",
    )
    assert passed, f"LLM judge failed. Response: {result}"
