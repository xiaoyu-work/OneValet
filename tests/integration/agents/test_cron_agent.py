"""Integration tests for CronAgent — tool selection, argument extraction, response quality.

CronAgent tools:
  cron_status, cron_list, cron_add, cron_update, cron_remove, cron_run, cron_runs
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.productivity]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Show me the cron status", ["cron_status"]),
    ("List all my scheduled jobs", ["cron_list"]),
    ("Schedule a cron job for daily briefing every morning at 8am", ["cron_add"]),
    ("Update the cron job Daily Briefing to run at 9am instead", ["cron_update"]),
    ("Delete the Weekly Report cron job", ["cron_remove"]),
    ("Trigger the Daily Briefing cron job right now", ["cron_run"]),
    ("Show me the run history for the Daily Briefing cron job", ["cron_runs"]),
    ("Alert me if Bitcoin drops below $50k, check every 5 minutes", ["cron_add"]),
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

async def test_cron_add_extracts_schedule(orchestrator_factory):
    """cron_add should receive a cron expression or schedule value for 'every morning at 8am'."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Schedule a reminder every morning at 8am to drink water")

    add_calls = [c for c in recorder.tool_calls if c["tool_name"] == "cron_add"]
    assert add_calls, "Expected cron_add to be called"

    args = add_calls[0]["arguments"]
    # The agent should set a cron-style schedule; value should contain "8" somewhere
    schedule_value = args.get("schedule_value", "") or args.get("schedule", "")
    assert schedule_value, f"Expected schedule_value in args, got {args}"


async def test_cron_add_conditional_flag(orchestrator_factory):
    """Conditional alerts ('alert me if X') should set conditional=True."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Alert me if Bitcoin drops below $50,000, check every hour",
    )

    add_calls = [c for c in recorder.tool_calls if c["tool_name"] == "cron_add"]
    assert add_calls, "Expected cron_add to be called"

    args = add_calls[0]["arguments"]
    # conditional should be True for "alert me if" patterns
    assert args.get("conditional") is True, (
        f"Expected conditional=True for alert pattern, got {args}"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_create_job(orchestrator_factory, llm_judge):
    """After creating a cron job, the response should confirm the schedule."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Set up a daily briefing every morning at 8am"
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Set up a daily briefing every morning at 8am",
        response=response,
        criteria=(
            "The response should confirm that a scheduled job was created. "
            "It should mention the schedule (daily/8am/morning) and not ask "
            "unnecessary follow-up questions."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"


async def test_response_quality_list_jobs(orchestrator_factory, llm_judge):
    """Listing jobs should present results clearly."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "What cron jobs do I have?")
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="What cron jobs do I have?",
        response=response,
        criteria=(
            "The response should list the user's cron jobs with names and schedules. "
            "It should be well-formatted and easy to read."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"
