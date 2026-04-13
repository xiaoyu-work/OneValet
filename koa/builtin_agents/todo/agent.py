"""
TodoAgent - Agent for all todo, reminder, and task management requests.

Replaces the separate TodoQueryAgent, CreateTodoAgent, UpdateTodoAgent, DeleteTodoAgent,
ReminderAgent, TaskManagementAgent, and PlannerAgent with a single agent that has its own
mini ReAct loop. The orchestrator sees only one "TodoAgent" tool instead of seven.

The internal LLM decides which tools to call (query_tasks, create_task, update_task,
delete_task, set_reminder, manage_reminders) based on the user's request.
"""

from koa import valet
from koa.standard_agent import StandardAgent

from ..shared.routing_preferences import set_routing_preference
from .tools import (
    check_overdue_tasks,
    create_task,
    delete_task,
    manage_reminders,
    query_tasks,
    remember_important_date,
    set_reminder,
    update_task,
)


@valet(domain="productivity")
class TodoAgent(StandardAgent):
    """List, create, complete, and delete todo tasks; set and manage reminders. Use when the user mentions tasks, todos, to-do lists, reminders, or wants to be reminded about something."""

    max_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a todo and reminder management assistant with access to task and reminder tools.

Available tools:
- query_tasks: List or search the user's todo tasks in the resolved destination.
- create_task: Create a new todo task with title, optional due date and priority.
- update_task: Mark a todo task as complete by searching for it.
- delete_task: Delete a todo task by searching for it.
- set_reminder: Create a time-based reminder (one-time or recurring).
- remember_important_date: Save durable dates like birthdays and anniversaries.
- manage_reminders: List, update, pause, resume, or delete scheduled reminders and automations.
- set_routing_preference: Save the user's default todo destination.

Today: {today} ({weekday}), current time: {current_time}, timezone: {timezone}

Instructions:
1. If the user says to remember a default destination for todos, call set_routing_preference with surface="todo".
2. For task queries (list, search), call query_tasks.
3. For creating tasks, call create_task with the title and any mentioned due date or priority.
4. For completing/marking done, call update_task with a search query describing the task.
5. For deleting tasks, call delete_task with a search query.
6. For time-based reminders ("remind me in 5 minutes", "every day at 8am"), call set_reminder. \
Calculate the exact schedule_datetime from the current date/time.
7. Use remember_important_date for durable dates like birthdays and anniversaries.
8. If the user explicitly names a target provider, pass target_provider/target_account.
9. For managing existing reminders ("show my reminders", "pause my morning alert", \
"delete my medicine reminder"), call manage_reminders with the appropriate action.
10. If the user's request is ambiguous or missing information, ask for clarification \
in your text response WITHOUT calling any tools.
11. After getting tool results, provide a clear summary to the user."""

    def get_system_prompt(self) -> str:
        now, tz_name = self._user_now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
            current_time=now.strftime("%H:%M"),
            timezone=tz_name,
        )

    tools = (
        query_tasks,
        create_task,
        update_task,
        delete_task,
        set_reminder,
        remember_important_date,
        manage_reminders,
        set_routing_preference,
        check_overdue_tasks,  # proactive/internal-only: called by the overdue-task cron job
    )
