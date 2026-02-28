"""Integration tests for TodoAgent.

Tests tool selection, argument extraction, and response quality for:
- query_tasks: List or search todo tasks across connected providers
- create_task: Create a new todo task
- update_task: Mark a task as complete
- delete_task: Delete a task by keyword search
- set_reminder: Create a time-based reminder (one-time or recurring)
- manage_reminders: List, update, pause, resume, or delete reminders
"""

import pytest

pytestmark = [pytest.mark.integration]


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
    ("Delete the groceries task", ["delete_task"]),
    ("Remove the call dentist todo", ["delete_task"]),
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

async def test_extracts_create_task_title(orchestrator_factory):
    """create_task should receive the correct title from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Add a task: buy groceries")

    create_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "create_task"
    ]
    assert create_calls, "create_task was never called"

    args = create_calls[0]["arguments"]
    title = args.get("title", "").lower()
    assert "groceries" in title or "buy" in title, (
        f"Expected title containing 'groceries', got '{title}'"
    )


async def test_extracts_create_task_due_date(orchestrator_factory):
    """create_task should extract a due date when one is mentioned."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user", "Create a task to submit the report by 2026-03-15"
    )

    create_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "create_task"
    ]
    assert create_calls, "create_task was never called"

    args = create_calls[0]["arguments"]
    due = args.get("due", "") or ""
    assert "2026-03-15" in due or "03-15" in due or "march" in due.lower(), (
        f"Expected due date referencing 2026-03-15, got '{due}'"
    )


async def test_extracts_reminder_message_and_time(orchestrator_factory):
    """set_reminder should receive the reminder message and a schedule_datetime."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user", "Remind me to take my medicine at 9pm tonight"
    )

    reminder_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "set_reminder"
    ]
    assert reminder_calls, "set_reminder was never called"

    args = reminder_calls[0]["arguments"]
    message = args.get("reminder_message", "").lower()
    assert "medicine" in message, (
        f"Expected reminder_message containing 'medicine', got '{message}'"
    )
    assert args.get("schedule_datetime"), "schedule_datetime should not be empty"


async def test_extracts_manage_reminders_action(orchestrator_factory):
    """manage_reminders should receive the correct action."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Show me all my reminders")

    manage_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "manage_reminders"
    ]
    assert manage_calls, "manage_reminders was never called"

    args = manage_calls[0]["arguments"]
    action = args.get("action", "").lower()
    assert action in ("list", "show"), (
        f"Expected action='list' or 'show', got '{action}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_list_tasks(orchestrator_factory, llm_judge):
    """Listing tasks should produce a readable, structured output."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Show my tasks")

    passed = await llm_judge(
        "Show my tasks",
        result,
        "The response should present a list of tasks with titles and optionally "
        "their status or due dates. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result}"


async def test_response_quality_create_task(orchestrator_factory, llm_judge):
    """Creating a task should confirm the title and any due date."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Add a task to pick up dry cleaning by Friday"
    )

    passed = await llm_judge(
        "Add a task to pick up dry cleaning by Friday",
        result,
        "The response should confirm that a task was created with a title related "
        "to 'dry cleaning'. It should acknowledge the task creation positively.",
    )
    assert passed, f"LLM judge failed. Response: {result}"


async def test_response_quality_set_reminder(orchestrator_factory, llm_judge):
    """Setting a reminder should confirm the time and message."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Remind me to call mom tomorrow at 10am"
    )

    passed = await llm_judge(
        "Remind me to call mom tomorrow at 10am",
        result,
        "The response should confirm that a reminder has been set for approximately "
        "10am tomorrow. It should mention 'call mom' or the reminder content.",
    )
    assert passed, f"LLM judge failed. Response: {result}"
