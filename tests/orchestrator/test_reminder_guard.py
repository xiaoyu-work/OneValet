"""Tests for koa.orchestrator.reminder_guard."""

from __future__ import annotations

import pytest

from koa.orchestrator.react_config import ToolCallRecord
from koa.orchestrator.reminder_guard import (
    UNSCHEDULED_REMINDER_NOTE,
    reminder_guard_hook,
)
from koa.result import AgentResult, AgentStatus


def _make_result(text: str) -> AgentResult:
    return AgentResult(
        agent_type="Test",
        status=AgentStatus.COMPLETED,
        raw_message=text,
    )


def _ctx(tool_calls):
    return {"tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Base behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_message_is_passthrough():
    result = _make_result("")
    out = await reminder_guard_hook(result, _ctx([]))
    assert out.raw_message == ""


@pytest.mark.asyncio
async def test_no_reminder_commitment_is_passthrough():
    result = _make_result("Here's the weather for tomorrow: sunny.")
    out = await reminder_guard_hook(result, _ctx([]))
    assert out.raw_message == "Here's the weather for tomorrow: sunny."


@pytest.mark.asyncio
async def test_note_already_present_is_not_double_appended():
    original = f"I'll set a reminder for you.\n\n{UNSCHEDULED_REMINDER_NOTE}"
    result = _make_result(original)
    out = await reminder_guard_hook(result, _ctx([]))
    assert out.raw_message == original


# ---------------------------------------------------------------------------
# The historical bug: guard must recognize reminder tools and delegations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commitment_without_any_tool_triggers_note():
    result = _make_result("I'll set a reminder for tomorrow at 9am.")
    out = await reminder_guard_hook(result, _ctx([]))
    assert UNSCHEDULED_REMINDER_NOTE in out.raw_message


@pytest.mark.asyncio
async def test_direct_set_reminder_tool_suppresses_note():
    """Regression: ``set_reminder`` was missing from the allowlist."""
    tc = ToolCallRecord(
        name="set_reminder",
        args_summary={},
        result_status="COMPLETED",
        success=True,
    )
    result = _make_result("Reminder has been set for tomorrow at 9am.")
    out = await reminder_guard_hook(result, _ctx([tc]))
    assert UNSCHEDULED_REMINDER_NOTE not in out.raw_message


@pytest.mark.asyncio
async def test_direct_cron_add_suppresses_note():
    tc = ToolCallRecord(name="cron_add", args_summary={}, result_status="COMPLETED", success=True)
    result = _make_result("Reminder is scheduled for 8am daily.")
    out = await reminder_guard_hook(result, _ctx([tc]))
    assert UNSCHEDULED_REMINDER_NOTE not in out.raw_message


@pytest.mark.asyncio
async def test_todo_agent_completed_suppresses_note():
    """Regression: ``set_reminder`` runs inside TodoAgent's sub-ReAct loop,
    so only the ``TodoAgent`` name surfaces at the orchestrator level."""
    tc = ToolCallRecord(
        name="TodoAgent",
        args_summary={},
        result_status="COMPLETED",
        success=True,
    )
    result = _make_result("I'll remind you tomorrow at 9am: prepare interview questions.")
    out = await reminder_guard_hook(result, _ctx([tc]))
    assert UNSCHEDULED_REMINDER_NOTE not in out.raw_message


@pytest.mark.asyncio
async def test_todo_agent_waiting_for_approval_suppresses_note():
    tc = ToolCallRecord(
        name="TodoAgent",
        args_summary={},
        result_status="WAITING_FOR_APPROVAL",
        success=True,
    )
    result = _make_result("I'll set a reminder once you approve.")
    out = await reminder_guard_hook(result, _ctx([tc]))
    assert UNSCHEDULED_REMINDER_NOTE not in out.raw_message


@pytest.mark.asyncio
async def test_cron_agent_completed_suppresses_note():
    tc = ToolCallRecord(
        name="CronAgent",
        args_summary={},
        result_status="COMPLETED",
        success=True,
    )
    result = _make_result("Reminder has been created.")
    out = await reminder_guard_hook(result, _ctx([tc]))
    assert UNSCHEDULED_REMINDER_NOTE not in out.raw_message


@pytest.mark.asyncio
async def test_todo_agent_error_status_does_not_suppress_note():
    """If the delegated agent failed, we SHOULD warn the user."""
    tc = ToolCallRecord(
        name="TodoAgent",
        args_summary={},
        result_status="ERROR",
        success=False,
    )
    result = _make_result("I'll set a reminder for you tomorrow.")
    out = await reminder_guard_hook(result, _ctx([tc]))
    assert UNSCHEDULED_REMINDER_NOTE in out.raw_message


@pytest.mark.asyncio
async def test_unrelated_agent_does_not_suppress_note():
    """A non-reminder agent must not accidentally satisfy the guard."""
    tc = ToolCallRecord(
        name="EmailAgent",
        args_summary={},
        result_status="COMPLETED",
        success=True,
    )
    result = _make_result("I'll set a reminder for you tomorrow.")
    out = await reminder_guard_hook(result, _ctx([tc]))
    assert UNSCHEDULED_REMINDER_NOTE in out.raw_message


# ---------------------------------------------------------------------------
# Robustness: accept dict-shaped tool call records, lowercase statuses, etc.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dict_shaped_tool_call_is_accepted():
    result = _make_result("I'll set a reminder for tomorrow.")
    out = await reminder_guard_hook(
        result,
        _ctx([{"name": "set_reminder", "success": True, "result_status": "COMPLETED"}]),
    )
    assert UNSCHEDULED_REMINDER_NOTE not in out.raw_message


@pytest.mark.asyncio
async def test_result_status_case_is_normalized():
    """``result_status`` may arrive lowercase (AgentStatus enum value)."""
    tc = ToolCallRecord(
        name="TodoAgent",
        args_summary={},
        result_status="completed",  # lowercase
        success=True,
    )
    result = _make_result("I'll set a reminder for you.")
    out = await reminder_guard_hook(result, _ctx([tc]))
    assert UNSCHEDULED_REMINDER_NOTE not in out.raw_message
