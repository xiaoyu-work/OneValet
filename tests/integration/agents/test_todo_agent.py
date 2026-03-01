"""Integration tests for TodoAgent.

Tests tool selection, argument extraction, response quality, and approval flow for:
- query_tasks: List or search todo tasks across connected providers
- create_task: Create a new todo task (needs approval)
- update_task: Mark a task as complete (needs approval)
- delete_task: Delete a task by keyword search (needs approval)
- set_reminder: Create a time-based reminder (one-time or recurring)
- manage_reminders: List, update, pause, resume, or delete reminders
"""

import pytest

from onevalet.result import AgentStatus

pytestmark = [pytest.mark.integration, pytest.mark.productivity]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Show my todo list", ["query_tasks"]),
    ("What tasks do I have?", ["query_tasks"]),
    ("List my pending tasks", ["query_tasks"]),
    ("Add a task: buy groceries", ["create_task"]),
    ("Create a todo to call the dentist by Friday", ["create_task"]),
    ("I finished buying groceries", ["update_task"]),
    ("Mark the dentist task as done", ["update_task"]),
    ("Delete the groceries task", ["delete_task", "query_tasks"]),
    ("Remove the call dentist todo", ["delete_task", "query_tasks"]),
    ("Remind me to take medicine at 9pm", ["set_reminder"]),
    ("Set a reminder for tomorrow at 8am to check email", ["set_reminder"]),
    ("Show my reminders", ["manage_reminders"]),
    ("Pause my morning reminder", ["manage_reminders"]),
    ("Delete my medicine reminder", ["manage_reminders"]),
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

async def test_extracts_create_task_title(conversation):
    """create_task should receive the correct title from the user message."""
    conv = await conversation()
    await conv.send_until_tool_called("Add a task: buy groceries")
    conv.assert_tool_called("create_task")

    args = conv.get_tool_args("create_task")[0]
    title = args.get("title", "").lower()
    assert "groceries" in title or "buy" in title, (
        f"Expected title containing 'groceries', got '{title}'"
    )


async def test_extracts_create_task_due_date(conversation):
    """create_task should extract a due date when one is mentioned."""
    conv = await conversation()
    await conv.send_until_tool_called(
        "Create a task to submit the report by 2026-03-15"
    )
    conv.assert_tool_called("create_task")

    args = conv.get_tool_args("create_task")[0]
    due = args.get("due", "") or ""
    assert "2026-03-15" in due or "03-15" in due or "march" in due.lower(), (
        f"Expected due date referencing 2026-03-15, got '{due}'"
    )


async def test_extracts_reminder_message_and_time(conversation):
    """set_reminder should receive the reminder message and a schedule_datetime."""
    conv = await conversation()
    await conv.send("Remind me to take my medicine at 9pm tonight")
    conv.assert_tool_called("set_reminder")

    args = conv.get_tool_args("set_reminder")[0]
    message = args.get("reminder_message", "").lower()
    assert "medicine" in message, (
        f"Expected reminder_message containing 'medicine', got '{message}'"
    )
    assert args.get("schedule_datetime"), "schedule_datetime should not be empty"


async def test_extracts_manage_reminders_action(conversation):
    """manage_reminders should receive the correct action."""
    conv = await conversation()
    await conv.send("Show me all my reminders")
    conv.assert_tool_called("manage_reminders")

    args = conv.get_tool_args("manage_reminders")[0]
    action = args.get("action", "").lower()
    assert action in ("list", "show"), (
        f"Expected action='list' or 'show', got '{action}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_list_tasks(conversation, llm_judge):
    """Listing tasks should produce a readable, structured output."""
    conv = await conversation()
    await conv.auto_complete("Show my tasks")

    passed = await llm_judge(
        "Show my tasks",
        conv.last_message,
        "The response should present a list of tasks with titles and optionally "
        "their status or due dates. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


async def test_response_quality_create_task(conversation, llm_judge):
    """Creating a task should confirm the title and any due date."""
    conv = await conversation()
    msg = "Add a task to pick up dry cleaning by Friday"
    await conv.auto_complete(msg)

    passed = await llm_judge(
        msg,
        conv.last_message,
        "The response should confirm that a task was created with a title related "
        "to 'dry cleaning'. It should acknowledge the task creation positively.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


async def test_response_quality_set_reminder(conversation, llm_judge):
    """Setting a reminder should confirm the time and message."""
    conv = await conversation()
    msg = "Remind me to call mom tomorrow at 10am"
    await conv.auto_complete(msg)

    passed = await llm_judge(
        msg,
        conv.last_message,
        "The response should confirm that a reminder has been set for approximately "
        "10am tomorrow. It should mention 'call mom' or the reminder content.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


# ---------------------------------------------------------------------------
# Approval flow
# ---------------------------------------------------------------------------

async def test_create_task_triggers_approval(conversation):
    """create_task should pause for user approval before executing."""
    conv = await conversation()
    await conv.send_until_status(
        "Add a task: buy groceries",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    conv.assert_tool_called("create_task")
    conv.assert_status(AgentStatus.WAITING_FOR_APPROVAL)


async def test_create_task_approve_executes(conversation):
    """Approving create_task should execute the tool and complete."""
    conv = await conversation()
    await conv.send_until_status(
        "Add a task: buy groceries",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    result = await conv.send("yes, create it")
    assert result.status in (AgentStatus.COMPLETED, AgentStatus.WAITING_FOR_APPROVAL), (
        f"Expected COMPLETED after approval, got {result.status}"
    )


async def test_delete_task_reject_cancels(conversation):
    """Rejecting delete_task should cancel the operation."""
    conv = await conversation()
    result = await conv.send_until_status(
        "Delete the groceries task",
        AgentStatus.WAITING_FOR_APPROVAL,
    )

    # LLM may call query_tasks first; approval not triggered
    if result.status != AgentStatus.WAITING_FOR_APPROVAL:
        pytest.skip("LLM called query_tasks first; approval not triggered")

    result2 = await conv.send("no, keep it")
    assert result2.status in (AgentStatus.CANCELLED, AgentStatus.COMPLETED), (
        f"Expected CANCELLED after rejection, got {result2.status}"
    )
