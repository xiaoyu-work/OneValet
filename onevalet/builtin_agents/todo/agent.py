"""
TodoAgent - Agent for all todo, reminder, and task management requests.

Replaces the separate TodoQueryAgent, CreateTodoAgent, UpdateTodoAgent, DeleteTodoAgent,
ReminderAgent, TaskManagementAgent, and PlannerAgent with a single agent that has its own
mini ReAct loop. The orchestrator sees only one "TodoAgent" tool instead of seven.

The internal LLM decides which tools to call (query_tasks, create_task, update_task,
delete_task, set_reminder, manage_reminders) based on the user's request.
"""

from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent

from .tools import (
    query_tasks,
    create_task,
    update_task,
    delete_task,
    set_reminder,
    manage_reminders,
)


@valet(capabilities=["todo", "reminder", "task"])
class TodoAgent(StandardAgent):
    """List, create, complete, and delete todo tasks; set and manage reminders. Use when the user mentions tasks, todos, to-do lists, reminders, or wants to be reminded about something."""

    max_turns = 5

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

    tools = (query_tasks, create_task, update_task, delete_task, set_reminder, manage_reminders)
