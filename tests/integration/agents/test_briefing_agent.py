"""Integration tests for BriefingAgent.

Tests tool selection, argument extraction, and response quality for:
- get_briefing: Generate an on-demand daily briefing
- setup_daily_briefing: Schedule a recurring daily briefing cron job
- manage_briefing: Check status, enable, disable, or delete the briefing job
"""

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Give me my daily briefing", ["get_briefing"]),
    ("What's on my plate today?", ["get_briefing"]),
    ("Summarize my day", ["get_briefing"]),
    ("What do I have going on today?", ["get_briefing"]),
    ("Set up a daily briefing at 7am", ["setup_daily_briefing"]),
    ("Send me a morning summary every day at 8:00", ["setup_daily_briefing"]),
    ("Check the status of my daily briefing", ["manage_briefing"]),
    ("Pause my morning briefing", ["manage_briefing"]),
    ("Disable my daily digest", ["manage_briefing"]),
    ("Cancel my daily briefing", ["manage_briefing"]),
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

async def test_extracts_schedule_time(conversation):
    """setup_daily_briefing should receive the correct time."""
    conv = await conversation()
    await conv.send("Set up a daily briefing at 7:30 AM")
    conv.assert_tool_called("setup_daily_briefing")

    args = conv.get_tool_args("setup_daily_briefing")[0]
    schedule_time = args.get("schedule_time", "")
    # Accept "07:30" or "7:30"
    assert "7" in schedule_time and "30" in schedule_time, (
        f"Expected schedule_time containing 7:30, got '{schedule_time}'"
    )


async def test_extracts_manage_action_disable(conversation):
    """manage_briefing should receive action='disable' when user asks to pause."""
    conv = await conversation()
    await conv.send("Pause my daily briefing")
    conv.assert_tool_called("manage_briefing")

    args = conv.get_tool_args("manage_briefing")[0]
    action = args.get("action", "").lower()
    assert action in ("disable", "pause"), (
        f"Expected action='disable' or 'pause', got '{action}'"
    )


async def test_extracts_manage_action_status(conversation):
    """manage_briefing should receive action='status' for a status check."""
    conv = await conversation()
    await conv.send("What is the status of my daily briefing?")
    conv.assert_tool_called("manage_briefing")

    args = conv.get_tool_args("manage_briefing")[0]
    assert args.get("action", "").lower() == "status"


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_briefing(conversation, llm_judge):
    """On-demand briefing should present calendar, tasks, and emails clearly."""
    conv = await conversation()
    await conv.auto_complete("Give me my morning briefing")

    passed = await llm_judge(
        "Give me my morning briefing",
        conv.last_message,
        "The response should present a daily briefing that organizes information "
        "from calendar events, tasks, and/or emails. It should be structured and "
        "readable, not an error message.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


async def test_response_quality_schedule(conversation, llm_judge):
    """Scheduling a briefing should confirm the time and recurrence."""
    conv = await conversation()
    msg = "Schedule a daily briefing for 8am"
    await conv.auto_complete(msg)

    passed = await llm_judge(
        msg,
        conv.last_message,
        "The response should confirm that a daily briefing has been scheduled at "
        "or around 8:00 AM. It should mention the schedule or timing.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"
