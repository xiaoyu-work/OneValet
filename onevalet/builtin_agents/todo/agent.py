"""
TodoAgent - Domain agent for all todo, reminder, and task management requests.

Replaces the separate TodoQueryAgent, CreateTodoAgent, UpdateTodoAgent, DeleteTodoAgent,
ReminderAgent, TaskManagementAgent, and PlannerAgent with a single agent that has its own
mini ReAct loop. The orchestrator sees only one "TodoAgent" tool instead of seven.

The internal LLM decides which tools to call (query_tasks, create_task, update_task,
delete_task, set_reminder, manage_reminders) based on the user's request.
"""

from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent, AgentTool

from .tools import (
    query_tasks,
    create_task,
    update_task,
    delete_task,
    set_reminder,
    manage_reminders,
)


# =============================================================================
# Approval preview functions (must be defined before the class body references them)
# =============================================================================

async def _create_task_preview(args: dict, context) -> str:
    title = args.get("title", "")
    due = args.get("due", "")
    priority = args.get("priority", "")
    parts = [f"Create task: {title}"]
    if due:
        parts.append(f"Due: {due}")
    if priority:
        parts.append(f"Priority: {priority}")
    parts.append("\nCreate this task?")
    return "\n".join(parts)


async def _update_task_preview(args: dict, context) -> str:
    search_query = args.get("search_query", "")
    indices = args.get("task_indices")
    if indices:
        return f"Mark task(s) #{', #'.join(str(i) for i in indices)} as complete?"
    return f"Search for and complete task matching: \"{search_query}\"?"


async def _delete_task_preview(args: dict, context) -> str:
    search_query = args.get("search_query", "")
    indices = args.get("task_indices")
    if indices:
        return f"Delete task(s) #{', #'.join(str(i) for i in indices)}?"
    return f"Search for and delete task matching: \"{search_query}\"?"


@valet(capabilities=["todo", "reminder", "task"])
class TodoAgent(StandardAgent):
    """List, create, complete, and delete todo tasks; set and manage reminders. Use when the user mentions tasks, todos, to-do lists, reminders, or wants to be reminded about something."""

    max_domain_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a todo and reminder management assistant with access to task and reminder tools.

Available tools:
- query_tasks: List or search the user's todo tasks across all connected providers.
- create_task: Create a new todo task with title, optional due date and priority.
- update_task: Mark a todo task as complete by searching for it.
- delete_task: Delete a todo task by searching for it.
- set_reminder: Create a time-based reminder (one-time or recurring).
- manage_reminders: List, update, pause, resume, or delete scheduled reminders and automations.

Today's date: {today} ({weekday})

Instructions:
1. For task queries (list, search), call query_tasks.
2. For creating tasks, call create_task with the title and any mentioned due date or priority.
3. For completing/marking done, call update_task with a search query describing the task.
4. For deleting tasks, call delete_task with a search query.
5. For time-based reminders ("remind me in 5 minutes", "every day at 8am"), call set_reminder. \
Calculate the exact schedule_datetime from the current date/time.
6. For managing existing reminders ("show my reminders", "pause my morning alert", \
"delete my medicine reminder"), call manage_reminders with the appropriate action.
7. If the user's request is ambiguous or missing information, ask for clarification \
in your text response WITHOUT calling any tools.
8. After getting tool results, provide a clear summary to the user."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )

    domain_tools = [
        AgentTool(
            name="query_tasks",
            description="List or search the user's todo tasks across all connected providers (Todoist, Google Tasks, Microsoft To Do).",
            parameters={
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Keywords to search for specific tasks. Omit or leave empty to list all pending tasks.",
                    },
                    "show_completed": {
                        "type": "boolean",
                        "description": "Whether to include completed tasks (default false).",
                    },
                },
                "required": [],
            },
            executor=query_tasks,
        ),
        AgentTool(
            name="create_task",
            description="Create a new todo task on the user's connected provider.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The task title or what needs to be done.",
                    },
                    "due": {
                        "type": "string",
                        "description": "Due date in YYYY-MM-DD format (optional).",
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority level: low, medium, high, or urgent (optional).",
                    },
                    "account": {
                        "type": "string",
                        "description": "Todo account name if the user specifies one (optional, defaults to primary).",
                    },
                },
                "required": ["title"],
            },
            executor=create_task,
            needs_approval=True,
            get_preview=_create_task_preview,
        ),
        AgentTool(
            name="update_task",
            description="Mark a todo task as complete by searching for it. Returns task list if multiple matches found.",
            parameters={
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Keywords to find the task to complete.",
                    },
                    "task_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "1-based indices of tasks to complete (use after seeing search results with multiple matches).",
                    },
                },
                "required": ["search_query"],
            },
            executor=update_task,
            needs_approval=True,
            get_preview=_update_task_preview,
        ),
        AgentTool(
            name="delete_task",
            description="Delete a todo task by searching for it. Returns task list if multiple matches found.",
            parameters={
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Keywords to find the task to delete.",
                    },
                    "task_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "1-based indices of tasks to delete (use after seeing search results with multiple matches).",
                    },
                },
                "required": ["search_query"],
            },
            executor=delete_task,
            needs_approval=True,
            get_preview=_delete_task_preview,
        ),
        AgentTool(
            name="set_reminder",
            description="Create a time-based reminder (one-time or recurring) via TriggerEngine.",
            parameters={
                "type": "object",
                "properties": {
                    "schedule_datetime": {
                        "type": "string",
                        "description": "ISO 8601 datetime when the reminder should fire (e.g., 2024-01-15T14:30:00).",
                    },
                    "schedule_type": {
                        "type": "string",
                        "enum": ["one_time", "recurring"],
                        "description": "Whether this is a one-time or recurring reminder.",
                    },
                    "cron_expression": {
                        "type": "string",
                        "description": "Cron expression for recurring reminders (e.g., '0 8 * * *' for daily 8am).",
                    },
                    "reminder_message": {
                        "type": "string",
                        "description": "What to remind the user about.",
                    },
                    "human_readable_time": {
                        "type": "string",
                        "description": "Friendly time description (e.g., 'in 5 minutes', 'every day at 8am').",
                    },
                },
                "required": ["schedule_datetime", "reminder_message"],
            },
            executor=set_reminder,
        ),
        AgentTool(
            name="manage_reminders",
            description="List, show details, update, pause, resume, or delete scheduled reminders and automations.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "show", "update", "pause", "resume", "delete"],
                        "description": "What to do with the reminder/automation.",
                    },
                    "task_hint": {
                        "type": "string",
                        "description": "Keywords to identify which reminder/automation (for show/update/pause/resume/delete).",
                    },
                    "status_filter": {
                        "type": "string",
                        "enum": ["active", "paused", "all"],
                        "description": "Filter by status when listing (default 'all').",
                    },
                    "new_schedule_datetime": {
                        "type": "string",
                        "description": "New ISO 8601 datetime for update action.",
                    },
                    "new_cron_expression": {
                        "type": "string",
                        "description": "New cron expression for update action.",
                    },
                    "new_message": {
                        "type": "string",
                        "description": "New reminder message for update action.",
                    },
                    "human_readable_time": {
                        "type": "string",
                        "description": "Friendly time description for update action.",
                    },
                    "update_type": {
                        "type": "string",
                        "enum": ["time", "message", "both"],
                        "description": "What to update: time, message, or both.",
                    },
                },
                "required": ["action"],
            },
            executor=manage_reminders,
        ),
    ]
