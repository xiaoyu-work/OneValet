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

async def test_extracts_schedule_time(orchestrator_factory):
    """setup_daily_briefing should receive the correct time."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Set up a daily briefing at 7:30 AM")

    setup_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "setup_daily_briefing"
    ]
    assert setup_calls, "setup_daily_briefing was never called"

    args = setup_calls[0]["arguments"]
    schedule_time = args.get("schedule_time", "")
    # Accept "07:30" or "7:30"
    assert "7" in schedule_time and "30" in schedule_time, (
        f"Expected schedule_time containing 7:30, got '{schedule_time}'"
    )


async def test_extracts_manage_action_disable(orchestrator_factory):
    """manage_briefing should receive action='disable' when user asks to pause."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Pause my daily briefing")

    manage_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "manage_briefing"
    ]
    assert manage_calls, "manage_briefing was never called"

    args = manage_calls[0]["arguments"]
    action = args.get("action", "").lower()
    assert action in ("disable", "pause"), (
        f"Expected action='disable' or 'pause', got '{action}'"
    )


async def test_extracts_manage_action_status(orchestrator_factory):
    """manage_briefing should receive action='status' for a status check."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user", "What is the status of my daily briefing?"
    )

    manage_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "manage_briefing"
    ]
    assert manage_calls, "manage_briefing was never called"

    args = manage_calls[0]["arguments"]
    assert args.get("action", "").lower() == "status"


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_briefing(orchestrator_factory, llm_judge):
    """On-demand briefing should present calendar, tasks, and emails clearly."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Give me my morning briefing")

    passed = await llm_judge(
        "Give me my morning briefing",
        result,
        "The response should present a daily briefing that organizes information "
        "from calendar events, tasks, and/or emails. It should be structured and "
        "readable, not an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result}"


async def test_response_quality_schedule(orchestrator_factory, llm_judge):
    """Scheduling a briefing should confirm the time and recurrence."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Schedule a daily briefing for 8am"
    )

    passed = await llm_judge(
        "Schedule a daily briefing for 8am",
        result,
        "The response should confirm that a daily briefing has been scheduled at "
        "or around 8:00 AM. It should mention the schedule or timing.",
    )
    assert passed, f"LLM judge failed. Response: {result}"
