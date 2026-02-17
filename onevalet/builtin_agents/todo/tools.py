"""
Todo Domain Tools - Standalone API functions for TodoAgent's mini ReAct loop.

Extracted from TodoQueryAgent, CreateTodoAgent, UpdateTodoAgent, DeleteTodoAgent,
ReminderAgent, TaskManagementAgent, and PlannerAgent.
Each function takes (args: dict, context: AgentToolContext) -> str.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from onevalet.standard_agent import AgentToolContext

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


def _get_task_repo(context: AgentToolContext):
    """Get TaskRepository from context hints."""
    from onevalet.builtin_agents.reminder.task_repo import TaskRepository
    db = context.context_hints.get("db") if context.context_hints else None
    if not db:
        return None
    return TaskRepository(db)


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

async def query_tasks(args: dict, context: AgentToolContext) -> str:
    """Query and search todo tasks across all connected providers."""
    search_query = args.get("search_query")
    show_completed = args.get("show_completed", False)

    try:
        accounts = await _resolve_accounts(context.tenant_id)

        if not accounts:
            return "No todo accounts found. Please connect one first."

        all_tasks = []
        failed_accounts = []

        # Skip meta keywords that mean "list all"
        meta_keywords = {"todo", "todos", "tasks", "task", "my tasks", "all", "list", "pending"}
        if search_query and search_query.lower() in meta_keywords:
            search_query = None

        for account in accounts:
            provider = _get_provider(account)
            if not provider:
                failed_accounts.append(account.get("email") or account.get("account_name", "unknown"))
                continue

            if not await provider.ensure_valid_token():
                failed_accounts.append(account.get("email") or account.get("account_name", "unknown"))
                continue

            try:
                if search_query:
                    result = await provider.search_tasks(query=search_query)
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

async def create_task(args: dict, context: AgentToolContext) -> str:
    """Create a new todo task on the user's connected provider."""
    title = args.get("title", "")
    due = args.get("due")
    priority = args.get("priority")
    account_spec = args.get("account", "primary")

    if not title:
        return "Error: task title is required."

    try:
        account = await _resolve_single_account(context.tenant_id, account_spec)
        if not account:
            return "I couldn't find your todo account. Please connect one in settings."

        provider = _get_provider(account)
        if not provider:
            return "Sorry, I can't create tasks with that provider yet."

        if not await provider.ensure_valid_token():
            return "I lost access to your todo account. Please reconnect it in settings."

        result = await provider.create_task(title=title, due=due, priority=priority)

        if result.get("success"):
            account_name = account.get("account_name", account.get("provider", ""))
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

async def update_task(args: dict, context: AgentToolContext) -> str:
    """Complete (mark as done) a todo task by searching for it."""
    search_query = args.get("search_query", "")
    task_indices = args.get("task_indices")  # Optional: pre-selected indices

    if not search_query:
        return "Error: search_query is required to find the task to complete."

    try:
        accounts = await _resolve_accounts(context.tenant_id)
        if not accounts:
            return "No todo accounts found. Please connect one first."

        all_tasks = []
        for account in accounts:
            provider = _get_provider(account)
            if not provider or not await provider.ensure_valid_token():
                continue
            try:
                result = await provider.search_tasks(query=search_query)
                if result.get("success"):
                    tasks = result.get("data", [])
                    for task in tasks:
                        task["_provider"] = account.get("provider", "")
                        task["_account_name"] = account.get("account_name", "")
                        task["_account_email"] = account.get("email", "")
                    all_tasks.extend(tasks)
            except Exception as e:
                logger.error(f"Failed to search {account.get('account_name')}: {e}", exc_info=True)

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
            account = await _resolve_single_account(context.tenant_id, email or "primary")
            if not account:
                failed_count += len(tasks)
                continue
            provider = _get_provider(account)
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

async def delete_task(args: dict, context: AgentToolContext) -> str:
    """Delete a todo task by searching for it."""
    search_query = args.get("search_query", "")
    task_indices = args.get("task_indices")  # Optional: pre-selected indices

    if not search_query:
        return "Error: search_query is required to find the task to delete."

    try:
        accounts = await _resolve_accounts(context.tenant_id)
        if not accounts:
            return "No todo accounts found. Please connect one first."

        all_tasks = []
        for account in accounts:
            provider = _get_provider(account)
            if not provider or not await provider.ensure_valid_token():
                continue
            try:
                result = await provider.search_tasks(query=search_query)
                if result.get("success"):
                    tasks = result.get("data", [])
                    for task in tasks:
                        task["_provider"] = account.get("provider", "")
                        task["_account_name"] = account.get("account_name", "")
                        task["_account_email"] = account.get("email", "")
                    all_tasks.extend(tasks)
            except Exception as e:
                logger.error(f"Failed to search {account.get('account_name')}: {e}", exc_info=True)

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
            account = await _resolve_single_account(context.tenant_id, email or "primary")
            if not account:
                failed_count += len(tasks)
                continue
            provider = _get_provider(account)
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

async def set_reminder(args: dict, context: AgentToolContext) -> str:
    """Create a reminder or scheduled automation via TriggerEngine."""
    schedule_datetime = args.get("schedule_datetime")
    schedule_type = args.get("schedule_type", "one_time")
    cron_expression = args.get("cron_expression")
    reminder_message = args.get("reminder_message", "")
    human_readable_time = args.get("human_readable_time", "")

    if not schedule_datetime:
        return "Error: schedule_datetime is required."
    if not reminder_message:
        return "Error: reminder_message is required."

    # Build trigger config
    try:
        if 'T' in schedule_datetime:
            local_dt = datetime.fromisoformat(schedule_datetime.replace('Z', '+00:00'))
        else:
            local_dt = datetime.fromisoformat(schedule_datetime)

        if schedule_type == "recurring" and cron_expression:
            trigger_config = {"cron": cron_expression}
        else:
            trigger_config = {"at": local_dt.isoformat()}
    except Exception as e:
        logger.error(f"Failed to parse schedule_datetime: {e}")
        return "I couldn't process that time. Could you try again?"

    # Get trigger engine from context_hints
    trigger_engine = context.context_hints.get("trigger_engine") if context.context_hints else None
    if not trigger_engine:
        return "Sorry, I can't create reminders right now. Please try again later."

    try:
        task = await trigger_engine.create_task(
            user_id=context.tenant_id,
            task_def={
                "name": reminder_message[:50],
                "description": f"Reminder: {reminder_message}",
                "trigger": {
                    "type": "schedule",
                    "config": trigger_config,
                },
                "action": {
                    "type": "notify",
                    "config": {
                        "message": f"Reminder: {reminder_message}",
                    },
                },
                "output": {
                    "channel": "sms",
                },
                "metadata": {
                    "created_by": "TodoAgent",
                    "human_readable_time": human_readable_time,
                },
            },
        )

        logger.info(f"Created reminder: {task.id}")
        time_desc = human_readable_time or schedule_datetime
        return f"Got it! I'll remind you {time_desc}: {reminder_message}"

    except Exception as e:
        logger.error(f"Failed to create reminder: {e}", exc_info=True)
        return "Sorry, I couldn't set up that reminder. Please try again."


# =============================================================================
# manage_reminders
# =============================================================================

async def manage_reminders(args: dict, context: AgentToolContext) -> str:
    """List, update, pause, resume, or delete scheduled reminders and automations."""
    action = args.get("action", "list")
    task_hint = args.get("task_hint", "")
    status_filter = args.get("status_filter", "all")

    repo = _get_task_repo(context)
    if not repo:
        return "Task storage is not available right now."

    trigger_engine = context.context_hints.get("trigger_engine") if context.context_hints else None

    if action == "list":
        return await _list_reminders(repo, context.tenant_id, status_filter)
    elif action == "show":
        return await _show_reminder(repo, context.tenant_id, task_hint)
    elif action == "pause":
        return await _pause_reminder(repo, context.tenant_id, task_hint, trigger_engine)
    elif action == "resume":
        return await _resume_reminder(repo, context.tenant_id, task_hint, trigger_engine)
    elif action == "delete":
        return await _delete_reminder(repo, context.tenant_id, task_hint, trigger_engine)
    elif action == "update":
        new_schedule_datetime = args.get("new_schedule_datetime")
        new_cron_expression = args.get("new_cron_expression")
        new_message = args.get("new_message")
        human_readable_time = args.get("human_readable_time", "")
        update_type = args.get("update_type")
        return await _update_reminder(
            repo, context.tenant_id, task_hint, trigger_engine,
            new_schedule_datetime, new_cron_expression, new_message,
            human_readable_time, update_type,
        )
    else:
        return "I'm not sure what you want to do. Try 'show my reminders' or 'delete my medicine reminder'."


# ---- Reminder sub-actions ----

def _match_tasks(tasks: List[Dict], hint: str) -> List[Dict]:
    """Match tasks by hint keywords."""
    if not hint or not tasks:
        return []
    hint_lower = hint.lower()
    matched = []
    for task in tasks:
        name = (task.get("name") or "").lower()
        description = (task.get("description") or "").lower()
        action_config = task.get("action_config", {})
        message = (action_config.get("message") or "").lower()
        searchable = f"{name} {description} {message}"
        if hint_lower in searchable:
            matched.append(task)
        elif any(word in searchable for word in hint_lower.split()):
            matched.append(task)
    return matched


def _format_cron(cron_expr: str) -> str:
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


def _format_schedule(trigger_config: Dict) -> str:
    """Format trigger config to human-readable schedule."""
    if trigger_config.get("cron"):
        return _format_cron(trigger_config["cron"])
    elif trigger_config.get("at"):
        return _format_datetime_display(trigger_config["at"])
    return "scheduled"


async def _list_reminders(repo, tenant_id: str, status_filter: str) -> str:
    """List all user's reminders/automations."""
    status_param = None if status_filter == "all" else status_filter
    tasks = await repo.get_user_tasks(tenant_id, status=status_param)
    if not tasks:
        if status_filter != "all":
            return f"You don't have any {status_filter} reminders or automations."
        return "You don't have any scheduled reminders or automations yet."

    lines = []
    for i, task in enumerate(tasks, 1):
        name = task.get("name", "Unnamed")
        status = task.get("status", "unknown")
        trigger_config = task.get("trigger_config", {})
        schedule = _format_schedule(trigger_config)
        status_icon = "" if status == "active" else " (paused)" if status == "paused" else " (disabled)"
        lines.append(f"{i}. {name} - {schedule}{status_icon}")

    return f"Your {len(tasks)} reminder(s)/automation(s):\n" + "\n".join(lines)


async def _show_reminder(repo, tenant_id: str, task_hint: str) -> str:
    """Show details of a specific reminder."""
    if not task_hint:
        return "Which reminder or automation would you like to see?"
    tasks = await repo.get_user_tasks(tenant_id)
    matched = _match_tasks(tasks, task_hint)
    if not matched:
        return f"I couldn't find anything matching '{task_hint}'."
    if len(matched) > 1:
        return _format_disambiguation(matched, "Which one do you mean?")
    task = matched[0]
    name = task.get("name", "Unnamed")
    status = task.get("status", "unknown")
    action_type = task.get("action_type", "")
    run_count = task.get("run_count", 0)
    type_label = "Reminder" if action_type == "notify" else "Automation"
    trigger_config = task.get("trigger_config", {})
    schedule = _format_schedule(trigger_config)
    action_config = task.get("action_config", {})
    message = action_config.get("message", "")
    lines = [f"{type_label}: {name}", f"Schedule: {schedule}", f"Status: {status}", f"Runs: {run_count}"]
    if message:
        lines.append(f"Message: {message}")
    return "\n".join(lines)


async def _pause_reminder(repo, tenant_id: str, task_hint: str, trigger_engine) -> str:
    """Pause a reminder/automation."""
    if not task_hint:
        return "Which reminder or automation would you like to pause?"
    tasks = await repo.get_user_tasks(tenant_id, status="active")
    matched = _match_tasks(tasks, task_hint)
    if not matched:
        return f"I couldn't find an active item matching '{task_hint}'."
    if len(matched) > 1:
        return _format_disambiguation(matched, "Which one should I pause?")
    task = matched[0]
    task_id = task.get("id")
    task_name = task.get("name", "item")
    try:
        await repo.update_task(task_id, {"status": "paused"})
        if trigger_engine:
            await trigger_engine._teardown_task_trigger_by_id(task_id)
        return f"Paused '{task_name}'. Say 'resume' when you want it back."
    except Exception as e:
        logger.error(f"Failed to pause task: {e}")
        return "Sorry, I couldn't pause that."


async def _resume_reminder(repo, tenant_id: str, task_hint: str, trigger_engine) -> str:
    """Resume a paused reminder/automation."""
    if not task_hint:
        return "Which reminder or automation would you like to resume?"
    tasks = await repo.get_user_tasks(tenant_id, status="paused")
    matched = _match_tasks(tasks, task_hint)
    if not matched:
        return f"I couldn't find a paused item matching '{task_hint}'."
    if len(matched) > 1:
        return _format_disambiguation(matched, "Which one should I resume?")
    task = matched[0]
    task_id = task.get("id")
    task_name = task.get("name", "item")
    try:
        updated_task = await repo.update_task(task_id, {"status": "active"})
        if trigger_engine and updated_task:
            from onevalet.triggers.models import Task
            task_obj = Task.from_dict(updated_task)
            await trigger_engine._setup_task_trigger(task_obj)
        return f"Resumed '{task_name}'."
    except Exception as e:
        logger.error(f"Failed to resume task: {e}")
        return "Sorry, I couldn't resume that."


async def _delete_reminder(repo, tenant_id: str, task_hint: str, trigger_engine) -> str:
    """Delete a reminder/automation."""
    if not task_hint:
        return "Which reminder or automation would you like to delete?"
    tasks = await repo.get_user_tasks(tenant_id)
    matched = _match_tasks(tasks, task_hint)
    if not matched:
        return f"I couldn't find anything matching '{task_hint}'."
    if len(matched) > 1:
        return _format_disambiguation(matched, "Which one should I delete?")
    task = matched[0]
    task_id = task.get("id")
    task_name = task.get("name", "item")
    try:
        if trigger_engine:
            try:
                await trigger_engine.delete_task(task_id)
            except Exception as e:
                logger.warning(f"Failed to delete from trigger engine: {e}")
        await repo.delete_task(task_id)
        return f"Deleted '{task_name}'."
    except Exception as e:
        logger.error(f"Failed to delete task: {e}")
        return "Sorry, I couldn't delete that."


async def _update_reminder(
    repo, tenant_id: str, task_hint: str, trigger_engine,
    new_schedule_datetime, new_cron_expression, new_message,
    human_readable_time, update_type,
) -> str:
    """Update a reminder's schedule or message."""
    if not task_hint:
        return "Which reminder or automation would you like to update?"
    tasks = await repo.get_user_tasks(tenant_id)
    matched = _match_tasks(tasks, task_hint)
    if not matched:
        return f"I couldn't find anything matching '{task_hint}'."
    if len(matched) > 1:
        return _format_disambiguation(matched, "Which one should I update?")
    task = matched[0]
    task_id = task.get("id")
    task_name = task.get("name", "item")
    try:
        update_data = {}
        response_parts = []
        if update_type in ("time", "both"):
            trigger_config = None
            if new_cron_expression:
                trigger_config = {"cron": new_cron_expression}
            elif new_schedule_datetime:
                try:
                    if 'T' in new_schedule_datetime:
                        local_dt = datetime.fromisoformat(new_schedule_datetime.replace('Z', '+00:00'))
                    else:
                        local_dt = datetime.fromisoformat(new_schedule_datetime)
                    trigger_config = {"at": local_dt.isoformat()}
                except Exception:
                    pass
            if trigger_config:
                update_data["trigger_config"] = trigger_config
                if new_cron_expression:
                    update_data["trigger_type"] = "schedule"
                time_desc = human_readable_time or "new time"
                response_parts.append(f"changed to {time_desc}")

        if update_type in ("message", "both"):
            if new_message:
                action_config = task.get("action_config", {})
                action_config["message"] = new_message
                update_data["action_config"] = action_config
                update_data["name"] = new_message[:50]
                update_data["description"] = f"Reminder: {new_message}"
                response_parts.append("message updated")

        if not update_data:
            return "What would you like to change? (time, message, or both)"

        updated_task = await repo.update_task(task_id, update_data)

        if "trigger_config" in update_data and trigger_engine:
            await trigger_engine._teardown_task_trigger_by_id(task_id)
            if updated_task and updated_task.get("status") == "active":
                from onevalet.triggers.models import Task
                task_obj = Task.from_dict(updated_task)
                await trigger_engine._setup_task_trigger(task_obj)

        response = f"Updated '{task_name}'"
        if response_parts:
            response += f" - {', '.join(response_parts)}"
        return response + "."
    except Exception as e:
        logger.error(f"Failed to update task: {e}", exc_info=True)
        return "Sorry, I couldn't update that. Please try again."


def _format_disambiguation(tasks: List[Dict], prompt: str) -> str:
    """Format task list for disambiguation."""
    lines = [f"Found {len(tasks)} matches:"]
    for i, task in enumerate(tasks, 1):
        name = task.get("name", "Unnamed")
        trigger_config = task.get("trigger_config", {})
        schedule = _format_schedule(trigger_config)
        lines.append(f"{i}. {name}" + (f" ({schedule})" if schedule else ""))
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
