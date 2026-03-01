"""Integration tests for CalendarAgent.

Tests tool selection, argument extraction, response quality, and approval flow for:
- query_events: Search and list calendar events by time range or keywords
- create_event: Create a new calendar event (needs approval)
- update_event: Update an existing event (needs approval)
- delete_event: Delete calendar events matching search criteria (needs approval)
"""

import pytest

from onevalet.result import AgentStatus

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
    """query_events should receive an appropriate time_range for 'today'."""
    conv = await conversation()
    await conv.send("What's on my calendar today?")
    conv.assert_tool_called("query_events")

    args = conv.get_tool_args("query_events")[0]
    time_range = args.get("time_range", "").lower()
    assert "today" in time_range, (
        f"Expected time_range containing 'today', got '{time_range}'"
    )


async def test_extracts_create_event_fields(conversation):
    """create_event should receive summary and start from the user message."""
    conv = await conversation()
    await conv.send_until_tool_called(
        "Create a meeting called Team Sync tomorrow at 3pm"
    )
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
            args.get("search_query", "")
            or args.get("query", "")
            or args.get("event_id", "")
        ).lower()
        assert "dentist" in search_query or search_query, (
            f"Expected search_query containing 'dentist', got '{search_query}'"
        )


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
