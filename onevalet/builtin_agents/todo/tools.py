"""
Todo Tools - Standalone API functions for TodoAgent's mini ReAct loop.

Extracted from TodoQueryAgent, CreateTodoAgent, UpdateTodoAgent, DeleteTodoAgent,
ReminderAgent, TaskManagementAgent, and PlannerAgent.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from onevalet.tool_decorator import tool
from onevalet.models import AgentToolContext

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================

async def _resolve_accounts(tenant_id: str):
    """Resolve all todo accounts for a tenant."""
    from onevalet.providers.todo.resolver import TodoAccountResolver
    return await TodoAccountResolver.resolve_accounts(tenant_id, ["all"])


async def _resolve_single_account(tenant_id: str, account_spec: str = "primary"):
    """Resolve a single todo account."""
    from onevalet.providers.todo.resolver import TodoAccountResolver
    return await TodoAccountResolver.resolve_account(tenant_id, account_spec)


def _get_provider(account):
    """Create a todo provider for the given account."""
    from onevalet.providers.todo.factory import TodoProviderFactory
    return TodoProviderFactory.create_provider(account)



def _format_due_date(due_str: str) -> str:
    """Format due date string to short display format."""
    if not due_str:
        return ""
    try:
        from dateutil import parser as date_parser
        dt = date_parser.parse(due_str)
        now = datetime.now()
        if dt.year == now.year:
            return dt.strftime("%b %d").lstrip("0")
        else:
            return dt.strftime("%b %d, %Y").lstrip("0")
    except Exception:
        return due_str


# =============================================================================
# query_tasks
# =============================================================================

@tool
async def query_tasks(
    search_query: Annotated[Optional[str], "Keywords to search for specific tasks. Omit or leave empty to list all pending tasks."] = None,
    show_completed: Annotated[bool, "Whether to include completed tasks (default false)."] = False,
    *,
    context: AgentToolContext,
) -> str:
    """List or search the user's todo tasks across all connected providers (Todoist, Google Tasks, Microsoft To Do)."""
    try:
        accounts = await _resolve_accounts(context.tenant_id)

        if not accounts:
            return "No todo accounts found. Please connect one first."

        all_tasks = []
        failed_accounts = []

        # Skip meta keywords that mean "list all"
        meta_keywords = {"todo", "todos", "tasks", "task", "my tasks", "all", "list", "pending"}
        effective_query = search_query
        if effective_query and effective_query.lower() in meta_keywords:
            effective_query = None

        for account in accounts:
            provider = _get_provider(account)
            if not provider:
                failed_accounts.append(account.get("email") or account.get("account_name", "unknown"))
                continue

            if not await provider.ensure_valid_token():
                failed_accounts.append(account.get("email") or account.get("account_name", "unknown"))
                continue

            try:
                if effective_query:
                    result = await provider.search_tasks(query=effective_query)
                else:
                    result = await provider.list_tasks(completed=show_completed)

                if result.get("success"):
                    tasks = result.get("data", [])
                    for task in tasks:
                        task["_provider"] = account.get("provider", "")
                        task["_account_name"] = account.get("account_name", "")
                        task["_account_email"] = account.get("email", "")
                    all_tasks.extend(tasks)

            except Exception as e:
                logger.error(f"Failed to query {account.get('account_name')}: {e}", exc_info=True)
                failed_accounts.append(account.get("email") or account.get("account_name", "unknown"))

        # Sort by due date (None dates last)
        all_tasks.sort(key=lambda t: t.get("due") or "9999-12-31")

        # Format output
        if not all_tasks and not failed_accounts:
            return "You're all caught up - no tasks found!"

        parts = []
        multi_provider = len(accounts) > 1

        if not all_tasks:
            parts.append("No tasks found.")
        else:
            parts.append(f"Found {len(all_tasks)} task(s):\n")
            for i, task in enumerate(all_tasks, 1):
                title = task.get("title", "Untitled")
                due = task.get("due")
                priority = task.get("priority")
                completed = task.get("completed", False)
                due_str = _format_due_date(due) if due else ""
                priority_str = ""
                if priority and priority.lower() not in ("none", "normal", "medium"):
                    priority_str = f" [{priority}]"
                check = "[x]" if completed else "[ ]"
                if multi_provider:
                    provider_name = task.get("_account_name", task.get("_provider", ""))
                    line = f"{i}. {check} [{provider_name}] {title}"
                else:
                    line = f"{i}. {check} {title}"
                if due_str:
                    line += f" - due {due_str}"
                if priority_str:
                    line += priority_str
                parts.append(line)

        for failed in failed_accounts:
            parts.append(f"\nCouldn't access {failed}. Please reconnect in settings.")

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Task search failed: {e}", exc_info=True)
        return "Couldn't search your tasks. Mind trying again later?"


# =============================================================================
# create_task
# =============================================================================

async def _preview_create_task(args: dict, context) -> str:
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


@tool(needs_approval=True, get_preview=_preview_create_task)
async def create_task(
    title: Annotated[str, "The task title or what needs to be done."],
    due: Annotated[Optional[str], "Due date in YYYY-MM-DD format (optional)."] = None,
    priority: Annotated[Optional[str], "Priority level: low, medium, high, or urgent (optional)."] = None,
    account: Annotated[str, "Todo account name if the user specifies one (optional, defaults to primary)."] = "primary",
    *,
    context: AgentToolContext,
) -> str:
    """Create a new todo task on the user's connected provider."""
    if not title:
        return "Error: task title is required."

    try:
        account_obj = await _resolve_single_account(context.tenant_id, account)
        if not account_obj:
            return "I couldn't find your todo account. Please connect one in settings."

        provider = _get_provider(account_obj)
        if not provider:
            return "Sorry, I can't create tasks with that provider yet."

        if not await provider.ensure_valid_token():
            return "I lost access to your todo account. Please reconnect it in settings."

        result = await provider.create_task(title=title, due=due, priority=priority)

        if result.get("success"):
            account_name = account_obj.get("account_name", account_obj.get("provider", ""))
            due_str = f" (due {due})" if due else ""
            return f"Added to {account_name}: {title}{due_str}"
        else:
            error_msg = result.get("error", "Unknown error")
            return f"Couldn't create the task: {error_msg}"

    except Exception as e:
        logger.error(f"Failed to create task: {e}", exc_info=True)
        return "Something went wrong creating your task. Want to try again?"


# =============================================================================
# update_task
# =============================================================================

async def _preview_update_task(args: dict, context) -> str:
    search_query = args.get("search_query", "")
    indices = args.get("task_indices")
    if indices:
        return f"Mark task(s) #{', #'.join(str(i) for i in indices)} as complete?"
    return f"Search for and complete task matching: \"{search_query}\"?"


@tool(needs_approval=True, get_preview=_preview_update_task)
async def update_task(
    search_query: Annotated[str, "Keywords to find the task to complete."],
    task_indices: Annotated[Optional[List[int]], "1-based indices of tasks to complete (use after seeing search results with multiple matches)."] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Mark a todo task as complete by searching for it. Returns task list if multiple matches found."""
    if not search_query:
        return "Error: search_query is required to find the task to complete."

    try:
        accounts = await _resolve_accounts(context.tenant_id)
        if not accounts:
            return "No todo accounts found. Please connect one first."

        all_tasks = []
        for account_obj in accounts:
            provider = _get_provider(account_obj)
            if not provider or not await provider.ensure_valid_token():
                continue
            try:
                result = await provider.search_tasks(query=search_query)
                if result.get("success"):
                    tasks = result.get("data", [])
                    for task in tasks:
                        task["_provider"] = account_obj.get("provider", "")
                        task["_account_name"] = account_obj.get("account_name", "")
                        task["_account_email"] = account_obj.get("email", "")
                    all_tasks.extend(tasks)
            except Exception as e:
                logger.error(f"Failed to search {account_obj.get('account_name')}: {e}", exc_info=True)

        # Fallback: list all tasks and filter with LLM
        if not all_tasks and context.llm_client:
            all_tasks = await _fallback_search(accounts, search_query, context.llm_client)

        if not all_tasks:
            return f"I couldn't find any tasks matching '{search_query}'."

        if len(all_tasks) > 1 and not task_indices:
            lines = [f"Found {len(all_tasks)} tasks matching '{search_query}':\n"]
            for i, task in enumerate(all_tasks[:10], 1):
                title = task.get("title", "Untitled")
                due = task.get("due", "")
                due_str = f" - due {due}" if due else ""
                account_name = task.get("_account_name", "")
                prefix = f"[{account_name}] " if account_name else ""
                lines.append(f"{i}. {prefix}{title}{due_str}")
            lines.append("\nPlease specify which task(s) to complete by calling update_task again with task_indices.")
            return "\n".join(lines)

        # Determine which tasks to complete
        tasks_to_complete = all_tasks
        if task_indices:
            tasks_to_complete = []
            for idx in task_indices:
                zero_idx = idx - 1
                if 0 <= zero_idx < len(all_tasks):
                    tasks_to_complete.append(all_tasks[zero_idx])

        if not tasks_to_complete:
            return "No valid tasks selected."

        # Complete tasks
        completed_count = 0
        failed_count = 0

        tasks_by_account: Dict[tuple, list] = {}
        for task in tasks_to_complete:
            key = (task.get("_provider", ""), task.get("_account_email", ""))
            tasks_by_account.setdefault(key, []).append(task)

        for (provider_name, email), tasks in tasks_by_account.items():
            account_obj = await _resolve_single_account(context.tenant_id, email or "primary")
            if not account_obj:
                failed_count += len(tasks)
                continue
            provider = _get_provider(account_obj)
            if not provider or not await provider.ensure_valid_token():
                failed_count += len(tasks)
                continue
            for task in tasks:
                try:
                    result = await provider.complete_task(
                        task_id=task.get("id", ""),
                        list_id=task.get("list_id")
                    )
                    if result.get("success"):
                        completed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Failed to complete task: {e}")
                    failed_count += 1

        if completed_count > 0 and failed_count == 0:
            if completed_count == 1:
                title = tasks_to_complete[0].get("title", "task")
                return f"Done! Marked \"{title}\" as complete."
            return f"Done! Completed {completed_count} task(s)."
        elif completed_count > 0:
            return f"Completed {completed_count} task(s), but {failed_count} failed."
        else:
            return "I had trouble completing those tasks. Want me to try again?"

    except Exception as e:
        logger.error(f"Failed to complete tasks: {e}", exc_info=True)
        return "Something went wrong. Want me to try again?"


# =============================================================================
# delete_task
# =============================================================================

async def _preview_delete_task(args: dict, context) -> str:
    search_query = args.get("search_query", "")
    indices = args.get("task_indices")
    if indices:
        return f"Delete task(s) #{', #'.join(str(i) for i in indices)}?"
    return f"Search for and delete task matching: \"{search_query}\"?"


@tool(needs_approval=True, get_preview=_preview_delete_task)
async def delete_task(
    search_query: Annotated[str, "Keywords to find the task to delete."],
    task_indices: Annotated[Optional[List[int]], "1-based indices of tasks to delete (use after seeing search results with multiple matches)."] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Delete a todo task by searching for it. Returns task list if multiple matches found."""
    if not search_query:
        return "Error: search_query is required to find the task to delete."

    try:
        accounts = await _resolve_accounts(context.tenant_id)
        if not accounts:
            return "No todo accounts found. Please connect one first."

        all_tasks = []
        for account_obj in accounts:
            provider = _get_provider(account_obj)
            if not provider or not await provider.ensure_valid_token():
                continue
            try:
                result = await provider.search_tasks(query=search_query)
                if result.get("success"):
                    tasks = result.get("data", [])
                    for task in tasks:
                        task["_provider"] = account_obj.get("provider", "")
                        task["_account_name"] = account_obj.get("account_name", "")
                        task["_account_email"] = account_obj.get("email", "")
                    all_tasks.extend(tasks)
            except Exception as e:
                logger.error(f"Failed to search {account_obj.get('account_name')}: {e}", exc_info=True)

        if not all_tasks and context.llm_client:
            all_tasks = await _fallback_search(accounts, search_query, context.llm_client)

        if not all_tasks:
            return f"I couldn't find any tasks matching '{search_query}'."

        if len(all_tasks) > 1 and not task_indices:
            lines = [f"Found {len(all_tasks)} tasks matching '{search_query}':\n"]
            for i, task in enumerate(all_tasks[:10], 1):
                title = task.get("title", "Untitled")
                due = task.get("due", "")
                due_str = f" - due {due}" if due else ""
                account_name = task.get("_account_name", "")
                prefix = f"[{account_name}] " if account_name else ""
                lines.append(f"{i}. {prefix}{title}{due_str}")
            lines.append("\nPlease specify which task(s) to delete by calling delete_task again with task_indices.")
            return "\n".join(lines)

        # Determine which tasks to delete
        tasks_to_delete = all_tasks
        if task_indices:
            tasks_to_delete = []
            for idx in task_indices:
                zero_idx = idx - 1
                if 0 <= zero_idx < len(all_tasks):
                    tasks_to_delete.append(all_tasks[zero_idx])

        if not tasks_to_delete:
            return "No valid tasks selected."

        # Delete tasks
        deleted_count = 0
        failed_count = 0

        tasks_by_account: Dict[tuple, list] = {}
        for task in tasks_to_delete:
            key = (task.get("_provider", ""), task.get("_account_email", ""))
            tasks_by_account.setdefault(key, []).append(task)

        for (provider_name, email), tasks in tasks_by_account.items():
            account_obj = await _resolve_single_account(context.tenant_id, email or "primary")
            if not account_obj:
                failed_count += len(tasks)
                continue
            provider = _get_provider(account_obj)
            if not provider or not await provider.ensure_valid_token():
                failed_count += len(tasks)
                continue
            for task in tasks:
                try:
                    result = await provider.delete_task(
                        task_id=task.get("id", ""),
                        list_id=task.get("list_id")
                    )
                    if result.get("success"):
                        deleted_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete task: {e}")
                    failed_count += 1

        if deleted_count > 0 and failed_count == 0:
            return f"Done! Deleted {deleted_count} task(s)."
        elif deleted_count > 0:
            return f"Deleted {deleted_count} task(s), but {failed_count} failed."
        else:
            return "I had trouble deleting those tasks. Want me to try again?"

    except Exception as e:
        logger.error(f"Failed to delete tasks: {e}", exc_info=True)
        return "Something went wrong. Want me to try again?"


# =============================================================================
# set_reminder
# =============================================================================

@tool
async def set_reminder(
    schedule_datetime: Annotated[str, "ISO 8601 datetime when the reminder should fire (e.g., 2024-01-15T14:30:00)."],
    reminder_message: Annotated[str, "What to remind the user about."],
    schedule_type: Annotated[str, "Whether this is a one-time or recurring reminder."] = "one_time",
    cron_expression: Annotated[Optional[str], "Cron expression for recurring reminders (e.g., '0 8 * * *' for daily 8am)."] = None,
    human_readable_time: Annotated[str, "Friendly time description (e.g., 'in 5 minutes', 'every day at 8am')."] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Create a time-based reminder (one-time or recurring) via CronService."""
    if not schedule_datetime:
        return "Error: schedule_datetime is required."
    if not reminder_message:
        return "Error: reminder_message is required."

    # Get cron service from context_hints
    cron_service = context.context_hints.get("cron_service") if context.context_hints else None
    if not cron_service:
        return "Sorry, I can't create reminders right now. Please try again later."

    from onevalet.triggers.cron.models import (
        AtSchedule, CronScheduleSpec,
        SessionTarget, WakeMode,
        AgentTurnPayload, DeliveryConfig, DeliveryMode,
        CronJobCreate,
    )

    # Build schedule
    try:
        if schedule_type == "recurring" and cron_expression:
            schedule = CronScheduleSpec(expr=cron_expression)
        else:
            if 'T' in schedule_datetime:
                local_dt = datetime.fromisoformat(schedule_datetime.replace('Z', '+00:00'))
            else:
                local_dt = datetime.fromisoformat(schedule_datetime)
            schedule = AtSchedule(at=local_dt.isoformat())
    except Exception as e:
        logger.error(f"Failed to parse schedule_datetime: {e}")
        return "I couldn't process that time. Could you try again?"

    try:
        job = await cron_service.add(CronJobCreate(
            name=reminder_message[:50],
            description=f"Reminder: {reminder_message}",
            user_id=context.tenant_id,
            schedule=schedule,
            session_target=SessionTarget.MAIN,
            wake_mode=WakeMode.NOW,
            payload=AgentTurnPayload(message=f"Reminder: {reminder_message}"),
            delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
            delete_after_run=isinstance(schedule, AtSchedule),
        ))

        logger.info(f"Created reminder via CronService: {job.id}")
        time_desc = human_readable_time or schedule_datetime
        return f"Got it! I'll remind you {time_desc}: {reminder_message}"

    except Exception as e:
        logger.error(f"Failed to create reminder: {e}", exc_info=True)
        return "Sorry, I couldn't set up that reminder. Please try again."


# =============================================================================
# manage_reminders
# =============================================================================

@tool
async def manage_reminders(
    action: Annotated[str, "What to do with the reminder/automation."],
    task_hint: Annotated[str, "Keywords to identify which reminder/automation (for show/update/pause/resume/delete)."] = "",
    status_filter: Annotated[str, "Filter by status when listing: 'all', 'active', 'paused' (default 'all')."] = "all",
    new_schedule_datetime: Annotated[Optional[str], "New ISO 8601 datetime for update action."] = None,
    new_cron_expression: Annotated[Optional[str], "New cron expression for update action."] = None,
    new_message: Annotated[Optional[str], "New reminder message for update action."] = None,
    human_readable_time: Annotated[str, "Friendly time description for update action."] = "",
    update_type: Annotated[Optional[str], "What to update: time, message, or both."] = None,
    *,
    context: AgentToolContext,
) -> str:
    """List, show details, update, pause, resume, or delete scheduled reminders and automations."""
    cron_service = context.context_hints.get("cron_service") if context.context_hints else None
    if not cron_service:
        return "Reminder service is not available right now."

    if action == "list":
        return _list_reminders(cron_service, context.tenant_id, status_filter)
    elif action == "show":
        return _show_reminder(cron_service, context.tenant_id, task_hint)
    elif action == "pause":
        return await _pause_reminder(cron_service, context.tenant_id, task_hint)
    elif action == "resume":
        return await _resume_reminder(cron_service, context.tenant_id, task_hint)
    elif action == "delete":
        return await _delete_reminder(cron_service, context.tenant_id, task_hint)
    elif action == "update":
        return await _update_reminder(
            cron_service, context.tenant_id, task_hint,
            new_schedule_datetime, new_cron_expression, new_message,
            human_readable_time, update_type,
        )
    else:
        return "I'm not sure what you want to do. Try 'show my reminders' or 'delete my medicine reminder'."


# ---- Reminder sub-actions (CronService-backed) ----

def _match_jobs(jobs, hint: str):
    """Match CronJob objects by hint keywords. Returns list of matching jobs."""
    if not hint or not jobs:
        return []
    hint_lower = hint.lower()
    matched = []
    for job in jobs:
        name = (job.name or "").lower()
        desc = (job.description or "").lower()
        payload_text = ""
        if hasattr(job.payload, "message"):
            payload_text = (job.payload.message or "").lower()
        elif hasattr(job.payload, "text"):
            payload_text = (job.payload.text or "").lower()
        searchable = f"{name} {desc} {payload_text}"
        if hint_lower in searchable:
            matched.append(job)
        elif any(word in searchable for word in hint_lower.split()):
            matched.append(job)
    return matched


def _format_cron_expr(cron_expr: str) -> str:
    """Convert cron expression to human-readable."""
    parts = cron_expr.split()
    if len(parts) != 5:
        return cron_expr
    minute, hour, day, month, weekday = parts
    if day == "*" and month == "*" and weekday == "*":
        return f"daily at {hour}:{minute.zfill(2)}"
    if day == "*" and month == "*" and weekday != "*":
        days = {"0": "Sun", "1": "Mon", "2": "Tue", "3": "Wed", "4": "Thu", "5": "Fri", "6": "Sat"}
        day_name = days.get(weekday, weekday)
        return f"every {day_name} at {hour}:{minute.zfill(2)}"
    return cron_expr


def _format_datetime_display(dt_str: str) -> str:
    """Format datetime for display."""
    try:
        if isinstance(dt_str, str):
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        else:
            dt = dt_str
        now = datetime.now(timezone.utc)
        if dt.date() == now.date():
            return f"today at {dt.strftime('%H:%M')}"
        elif (dt.date() - now.date()).days == 1:
            return f"tomorrow at {dt.strftime('%H:%M')}"
        else:
            return dt.strftime("%b %d at %H:%M")
    except Exception:
        return str(dt_str)[:16]


def _format_job_schedule(job) -> str:
    """Format a CronJob's schedule to human-readable string."""
    from onevalet.triggers.cron.models import AtSchedule, CronScheduleSpec, EverySchedule

    schedule = job.schedule
    if isinstance(schedule, CronScheduleSpec):
        return _format_cron_expr(schedule.expr)
    elif isinstance(schedule, AtSchedule):
        return _format_datetime_display(schedule.at)
    elif isinstance(schedule, EverySchedule):
        secs = schedule.every_ms / 1000
        if secs >= 3600:
            return f"every {secs / 3600:.0f}h"
        elif secs >= 60:
            return f"every {secs / 60:.0f}m"
        else:
            return f"every {secs:.0f}s"
    return "scheduled"


def _get_job_message(job) -> str:
    """Extract the message/text from a job's payload."""
    if hasattr(job.payload, "message"):
        return job.payload.message or ""
    elif hasattr(job.payload, "text"):
        return job.payload.text or ""
    return ""


def _list_reminders(cron_service, tenant_id: str, status_filter: str) -> str:
    """List all user's reminders/automations from CronService."""
    include_disabled = status_filter in ("all", "paused")
    jobs = cron_service.list_jobs(user_id=tenant_id, include_disabled=include_disabled)

    # Apply status filter
    if status_filter == "active":
        jobs = [j for j in jobs if j.enabled]
    elif status_filter == "paused":
        jobs = [j for j in jobs if not j.enabled]

    if not jobs:
        if status_filter != "all":
            return f"You don't have any {status_filter} reminders or automations."
        return "You don't have any scheduled reminders or automations yet."

    lines = []
    for i, job in enumerate(jobs, 1):
        name = job.name or "Unnamed"
        schedule_str = _format_job_schedule(job)
        status_icon = "" if job.enabled else " (paused)"
        lines.append(f"{i}. {name} - {schedule_str}{status_icon}")

    return f"Your {len(jobs)} reminder(s)/automation(s):\n" + "\n".join(lines)


def _show_reminder(cron_service, tenant_id: str, task_hint: str) -> str:
    """Show details of a specific reminder from CronService."""
    if not task_hint:
        return "Which reminder or automation would you like to see?"
    jobs = cron_service.list_jobs(user_id=tenant_id, include_disabled=True)
    matched = _match_jobs(jobs, task_hint)
    if not matched:
        return f"I couldn't find anything matching '{task_hint}'."
    if len(matched) > 1:
        return _format_job_disambiguation(matched, "Which one do you mean?")
    job = matched[0]
    schedule_str = _format_job_schedule(job)
    message = _get_job_message(job)
    status = "active" if job.enabled else "paused"
    last_status = job.state.last_run_status or "never run"
    lines = [
        f"Reminder: {job.name}",
        f"Schedule: {schedule_str}",
        f"Status: {status}",
        f"Last run: {last_status}",
    ]
    if job.state.consecutive_errors:
        lines.append(f"Consecutive errors: {job.state.consecutive_errors}")
    if message:
        lines.append(f"Message: {message}")
    if job.description:
        lines.append(f"Description: {job.description}")
    return "\n".join(lines)


async def _pause_reminder(cron_service, tenant_id: str, task_hint: str) -> str:
    """Pause a reminder/automation via CronService."""
    from onevalet.triggers.cron.models import CronJobPatch

    if not task_hint:
        return "Which reminder or automation would you like to pause?"
    jobs = cron_service.list_jobs(user_id=tenant_id, include_disabled=False)
    matched = _match_jobs(jobs, task_hint)
    if not matched:
        return f"I couldn't find an active item matching '{task_hint}'."
    if len(matched) > 1:
        return _format_job_disambiguation(matched, "Which one should I pause?")
    job = matched[0]
    try:
        await cron_service.update(job.id, CronJobPatch(enabled=False))
        return f"Paused '{job.name}'. Say 'resume' when you want it back."
    except Exception as e:
        logger.error(f"Failed to pause cron job: {e}")
        return "Sorry, I couldn't pause that."


async def _resume_reminder(cron_service, tenant_id: str, task_hint: str) -> str:
    """Resume a paused reminder/automation via CronService."""
    from onevalet.triggers.cron.models import CronJobPatch

    if not task_hint:
        return "Which reminder or automation would you like to resume?"
    jobs = cron_service.list_jobs(user_id=tenant_id, include_disabled=True)
    paused = [j for j in jobs if not j.enabled]
    matched = _match_jobs(paused, task_hint)
    if not matched:
        return f"I couldn't find a paused item matching '{task_hint}'."
    if len(matched) > 1:
        return _format_job_disambiguation(matched, "Which one should I resume?")
    job = matched[0]
    try:
        await cron_service.update(job.id, CronJobPatch(enabled=True))
        return f"Resumed '{job.name}'."
    except Exception as e:
        logger.error(f"Failed to resume cron job: {e}")
        return "Sorry, I couldn't resume that."


async def _delete_reminder(cron_service, tenant_id: str, task_hint: str) -> str:
    """Delete a reminder/automation via CronService."""
    if not task_hint:
        return "Which reminder or automation would you like to delete?"
    jobs = cron_service.list_jobs(user_id=tenant_id, include_disabled=True)
    matched = _match_jobs(jobs, task_hint)
    if not matched:
        return f"I couldn't find anything matching '{task_hint}'."
    if len(matched) > 1:
        return _format_job_disambiguation(matched, "Which one should I delete?")
    job = matched[0]
    try:
        await cron_service.remove(job.id)
        return f"Deleted '{job.name}'."
    except Exception as e:
        logger.error(f"Failed to delete cron job: {e}")
        return "Sorry, I couldn't delete that."


async def _update_reminder(
    cron_service, tenant_id: str, task_hint: str,
    new_schedule_datetime, new_cron_expression, new_message,
    human_readable_time, update_type,
) -> str:
    """Update a reminder's schedule or message via CronService."""
    from onevalet.triggers.cron.models import (
        AtSchedule, CronScheduleSpec, AgentTurnPayload,
        SystemEventPayload, CronJobPatch,
    )

    if not task_hint:
        return "Which reminder or automation would you like to update?"
    jobs = cron_service.list_jobs(user_id=tenant_id, include_disabled=True)
    matched = _match_jobs(jobs, task_hint)
    if not matched:
        return f"I couldn't find anything matching '{task_hint}'."
    if len(matched) > 1:
        return _format_job_disambiguation(matched, "Which one should I update?")
    job = matched[0]
    try:
        patch = CronJobPatch()
        response_parts = []

        if update_type in ("time", "both"):
            if new_cron_expression:
                patch.schedule = CronScheduleSpec(expr=new_cron_expression)
                patch.delete_after_run = False
                time_desc = human_readable_time or _format_cron_expr(new_cron_expression)
                response_parts.append(f"changed to {time_desc}")
            elif new_schedule_datetime:
                try:
                    if 'T' in new_schedule_datetime:
                        local_dt = datetime.fromisoformat(new_schedule_datetime.replace('Z', '+00:00'))
                    else:
                        local_dt = datetime.fromisoformat(new_schedule_datetime)
                    patch.schedule = AtSchedule(at=local_dt.isoformat())
                    patch.delete_after_run = True
                    time_desc = human_readable_time or _format_datetime_display(local_dt.isoformat())
                    response_parts.append(f"changed to {time_desc}")
                except Exception:
                    pass

        if update_type in ("message", "both"):
            if new_message:
                # Build new payload matching existing kind
                if isinstance(job.payload, SystemEventPayload):
                    patch.payload = SystemEventPayload(text=f"Reminder: {new_message}")
                else:
                    patch.payload = AgentTurnPayload(message=f"Reminder: {new_message}")
                patch.name = new_message[:50]
                patch.description = f"Reminder: {new_message}"
                response_parts.append("message updated")

        has_changes = any(
            getattr(patch, f) is not None
            for f in ("schedule", "payload", "name", "description", "delete_after_run")
        )
        if not has_changes:
            return "What would you like to change? (time, message, or both)"

        await cron_service.update(job.id, patch)

        response = f"Updated '{job.name}'"
        if response_parts:
            response += f" - {', '.join(response_parts)}"
        return response + "."
    except Exception as e:
        logger.error(f"Failed to update cron job: {e}", exc_info=True)
        return "Sorry, I couldn't update that. Please try again."


def _format_job_disambiguation(jobs, prompt: str) -> str:
    """Format CronJob list for disambiguation."""
    lines = [f"Found {len(jobs)} matches:"]
    for i, job in enumerate(jobs, 1):
        schedule_str = _format_job_schedule(job)
        lines.append(f"{i}. {job.name}" + (f" ({schedule_str})" if schedule_str else ""))
    lines.append(f"\n{prompt}")
    return "\n".join(lines)


# =============================================================================
# Shared fallback search helper
# =============================================================================

async def _fallback_search(accounts, search_query: str, llm_client) -> List[Dict]:
    """Fallback: list all tasks and filter with LLM."""
    all_tasks = []
    for account in accounts:
        provider = _get_provider(account)
        if not provider:
            continue
        if not await provider.ensure_valid_token():
            continue
        try:
            result = await provider.list_tasks(max_results=50)
            if result.get("success"):
                tasks = result.get("data", [])
                for task in tasks:
                    task["_provider"] = account.get("provider", "")
                    task["_account_name"] = account.get("account_name", "")
                    task["_account_email"] = account.get("email", "")
                all_tasks.extend(tasks)
        except Exception:
            continue

    if not all_tasks or not llm_client:
        return []

    task_list = [
        {"index": i, "title": t.get("title", ""), "due": t.get("due", "")}
        for i, t in enumerate(all_tasks)
    ]

    prompt = f"""Find tasks matching: "{search_query}"

Tasks: {json.dumps(task_list)}

Return a JSON array of matching indices (0-based), like: [0, 3, 5]"""

    try:
        result = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            enable_thinking=False,
        )
        match = re.search(r'\[[\d,\s]*\]', result.content)
        if match:
            indices = json.loads(match.group())
            return [all_tasks[i] for i in indices if 0 <= i < len(all_tasks)]
    except Exception as e:
        logger.error(f"Filter LLM failed: {e}")

    return []
