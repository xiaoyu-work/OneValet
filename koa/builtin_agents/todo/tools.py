"""
Todo Tools - Standalone API functions for TodoAgent's mini ReAct loop.

Extracted from TodoQueryAgent, CreateTodoAgent, UpdateTodoAgent, DeleteTodoAgent,
ReminderAgent, TaskManagementAgent, and PlannerAgent.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, List, Optional

from koa.builtin_agents.shared.routing_preferences import (
    resolve_surface_target,
    wrap_routing_error,
)
from koa.models import AgentToolContext, ToolOutput
from koa.providers.local_backend import LocalBackendClient
from koa.providers.todo.local import LocalTodoProvider
from koa.tool_decorator import tool

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================


def _get_provider(account):
    """Create a todo provider for the given account."""
    from koa.providers.todo.factory import TodoProviderFactory

    return TodoProviderFactory.create_provider(account)


async def _resolve_todo_provider(
    context: AgentToolContext,
    target_provider: str | None = None,
    target_account: str | None = None,
    *,
    operation: str = "write",
):
    from koa.providers.todo.factory import TodoProviderFactory
    from koa.providers.todo.resolver import TodoAccountResolver

    backend_client = LocalBackendClient.from_context(context)

    try:
        target = await resolve_surface_target(
            tenant_id=context.tenant_id,
            surface="todo",
            backend_client=backend_client,
            explicit_provider=target_provider,
            explicit_account=target_account,
        )
    except Exception as e:
        logger.error(f"Failed to resolve todo routing target: {e}", exc_info=True)
        error_reason = "read_failed" if operation == "read" else "write_failed"
        return None, None, wrap_routing_error("todo", target_provider or "local", error_reason)

    if target.provider == "local":
        return (
            LocalTodoProvider(context.tenant_id, backend_client),
            {"provider": "local", "account_name": "local", "email": ""},
            None,
        )

    if target.provider not in TodoProviderFactory.get_supported_providers():
        return None, None, wrap_routing_error("todo", target.provider, "unsupported_provider")

    account = await TodoAccountResolver.resolve_account_for_provider(
        context.tenant_id,
        target.provider,
        target.account,
    )
    if not account:
        return None, None, wrap_routing_error("todo", target.provider, "not_connected")

    provider = _get_provider(account)
    if not provider:
        return None, None, wrap_routing_error("todo", target.provider, "unsupported_provider")

    if not await provider.ensure_valid_token():
        return None, None, wrap_routing_error("todo", target.provider, "auth_expired")

    return provider, account, None


def _annotate_tasks(tasks: List[dict], account: dict) -> List[dict]:
    annotated = []
    for task in tasks:
        task_copy = dict(task)
        task_copy["_provider"] = account.get("provider", "")
        task_copy["_account_name"] = account.get("account_name", "")
        task_copy["_account_email"] = account.get("email", "")
        annotated.append(task_copy)
    return annotated


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


def _normalize_important_date(date_str: str) -> Optional[str]:
    """Normalize a durable date input to YYYY-MM-DD."""
    if not date_str:
        return None

    try:
        from dateutil import parser as date_parser

        default = datetime.now().replace(
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        return date_parser.parse(date_str, fuzzy=True, default=default).date().isoformat()
    except Exception:
        return None


# =============================================================================
# query_tasks
# =============================================================================


@tool
async def query_tasks(
    search_query: Annotated[
        Optional[str],
        "Keywords to search for specific tasks. Omit or leave empty to list all pending tasks.",
    ] = None,
    show_completed: Annotated[bool, "Whether to include completed tasks (default false)."] = False,
    target_provider: Annotated[
        Optional[str], "Optional explicit provider like local, google, or todoist."
    ] = None,
    target_account: Annotated[
        Optional[str], "Optional explicit account label like primary or work."
    ] = None,
    *,
    context: AgentToolContext,
) -> str:
    """List or search the user's todo tasks in the resolved destination."""
    try:
        provider, account, error = await _resolve_todo_provider(
            context,
            target_provider=target_provider,
            target_account=target_account,
            operation="read",
        )
        if error:
            return error

        meta_keywords = {"todo", "todos", "tasks", "task", "my tasks", "all", "list", "pending"}
        effective_query = search_query
        if effective_query and effective_query.lower() in meta_keywords:
            effective_query = None

        if effective_query:
            result = await provider.search_tasks(query=effective_query)
        else:
            result = await provider.list_tasks(completed=show_completed)

        if not result.get("success"):
            return wrap_routing_error("todo", account.get("provider", "todo"), "read_failed")

        all_tasks = _annotate_tasks(result.get("data", []), account)

        # Sort by due date (None dates last)
        all_tasks.sort(key=lambda t: t.get("due") or "9999-12-31")

        # Format output
        if not all_tasks:
            return "You're all caught up - no tasks found!"

        parts = []
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
            line = f"{i}. {check} {title}"
            if due_str:
                line += f" - due {due_str}"
            if priority_str:
                line += priority_str
            parts.append(line)

        text_result = "\n".join(parts)

        # Build inline cards for frontend rendering
        if all_tasks:
            task_cards = []
            for task in all_tasks:
                card = {
                    "card_type": "task_item",
                    "title": task.get("title", "Untitled"),
                    "completed": bool(task.get("completed", False)),
                    "provider": task.get("_account_name", task.get("_provider", "")),
                }
                due = task.get("due")
                if due:
                    card["dueDate"] = _format_due_date(due)
                priority = task.get("priority")
                if priority and priority.lower() not in ("none", "normal", "medium"):
                    card["priority"] = priority
                task_cards.append(card)

            media = [
                {
                    "type": "inline_cards",
                    "data": json.dumps(task_cards),
                    "media_type": "application/json",
                    "metadata": {"for_storage": False},
                }
            ]
            return ToolOutput(text=text_result, media=media)

        return text_result

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
    card = {
        "card_type": "task_draft",
        "title": title,
        "dueDate": due or "",
        "priority": priority or "",
        "options": ["approve", "edit", "decline"],
    }
    return "📝 **Task draft**\n\n<!-- inline_card:" + json.dumps(card, ensure_ascii=False) + " -->"


@tool(needs_approval=True, get_preview=_preview_create_task)
async def create_task(
    title: Annotated[str, "The task title or what needs to be done."],
    due: Annotated[Optional[str], "Due date in YYYY-MM-DD format (optional)."] = None,
    priority: Annotated[
        Optional[str], "Priority level: low, medium, high, or urgent (optional)."
    ] = None,
    account: Annotated[
        Optional[str], "Deprecated compatibility alias for target_account when the user specifies one."
    ] = None,
    target_provider: Annotated[
        Optional[str], "Optional explicit provider like local, google, or todoist."
    ] = None,
    target_account: Annotated[
        Optional[str], "Optional explicit account label like primary or work."
    ] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Create a new todo task in the resolved destination."""
    if not title:
        return "Error: task title is required."

    try:
        provider, account_obj, error = await _resolve_todo_provider(
            context,
            target_provider=target_provider,
            target_account=target_account or account,
        )
        if error:
            return error

        result = await provider.create_task(title=title, due=due, priority=priority)

        if result.get("success"):
            due_str = f" (due {due})" if due else ""
            return f"Done! I added '{title}' to your tasks{due_str}."
        else:
            return wrap_routing_error("todo", account_obj.get("provider", "todo"), "write_failed")

    except Exception as e:
        logger.error(f"Failed to create task: {e}", exc_info=True)
        return wrap_routing_error("todo", target_provider or "local", "write_failed")


# =============================================================================
# update_task
# =============================================================================


async def _preview_update_task(args: dict, context) -> str:
    search_query = args.get("search_query", "")
    indices = args.get("task_indices")
    if indices:
        return f"Mark task(s) #{', #'.join(str(i) for i in indices)} as complete?"
    return f'Search for and complete task matching: "{search_query}"?'


@tool(needs_approval=True, get_preview=_preview_update_task)
async def update_task(
    search_query: Annotated[str, "Keywords to find the task to complete."],
    task_indices: Annotated[
        Optional[List[int]],
        "1-based indices of tasks to complete (use after seeing search results with multiple matches).",
    ] = None,
    target_provider: Annotated[
        Optional[str], "Optional explicit provider like local, google, or todoist."
    ] = None,
    target_account: Annotated[
        Optional[str], "Optional explicit account label like primary or work."
    ] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Mark a todo task as complete by searching for it. Returns task list if multiple matches found."""
    if not search_query:
        return "Error: search_query is required to find the task to complete."

    try:
        provider, account_obj, error = await _resolve_todo_provider(
            context,
            target_provider=target_provider,
            target_account=target_account,
        )
        if error:
            return error

        result = await provider.search_tasks(query=search_query)
        if not result.get("success"):
            return wrap_routing_error("todo", account_obj.get("provider", "todo"), "read_failed")

        all_tasks = _annotate_tasks(result.get("data", []), account_obj)

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
            lines.append(
                "\nPlease specify which task(s) to complete by calling update_task again with task_indices."
            )
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

        for task in tasks_to_complete:
            try:
                result = await provider.complete_task(
                    task_id=task.get("id", ""),
                    list_id=task.get("list_id"),
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
                return f'Done! Marked "{title}" as complete.'
            return f"Done! Completed {completed_count} task(s)."
        elif completed_count > 0:
            return f"Completed {completed_count} task(s), but {failed_count} failed."
        else:
            return wrap_routing_error("todo", account_obj.get("provider", "todo"), "write_failed")

    except Exception as e:
        logger.error(f"Failed to complete tasks: {e}", exc_info=True)
        return wrap_routing_error("todo", target_provider or "local", "write_failed")


# =============================================================================
# delete_task
# =============================================================================


async def _preview_delete_task(args: dict, context) -> str:
    search_query = args.get("search_query", "")
    indices = args.get("task_indices")
    if indices:
        return f"Delete task(s) #{', #'.join(str(i) for i in indices)}?"
    return f'Search for and delete task matching: "{search_query}"?'


@tool(needs_approval=True, get_preview=_preview_delete_task)
async def delete_task(
    search_query: Annotated[str, "Keywords to find the task to delete."],
    task_indices: Annotated[
        Optional[List[int]],
        "1-based indices of tasks to delete (use after seeing search results with multiple matches).",
    ] = None,
    target_provider: Annotated[
        Optional[str], "Optional explicit provider like local, google, or todoist."
    ] = None,
    target_account: Annotated[
        Optional[str], "Optional explicit account label like primary or work."
    ] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Delete a todo task by searching for it. Returns task list if multiple matches found."""
    if not search_query:
        return "Error: search_query is required to find the task to delete."

    try:
        provider, account_obj, error = await _resolve_todo_provider(
            context,
            target_provider=target_provider,
            target_account=target_account,
        )
        if error:
            return error

        result = await provider.search_tasks(query=search_query)
        if not result.get("success"):
            return wrap_routing_error("todo", account_obj.get("provider", "todo"), "read_failed")

        all_tasks = _annotate_tasks(result.get("data", []), account_obj)

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
            lines.append(
                "\nPlease specify which task(s) to delete by calling delete_task again with task_indices."
            )
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

        for task in tasks_to_delete:
            try:
                result = await provider.delete_task(
                    task_id=task.get("id", ""),
                    list_id=task.get("list_id"),
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
            return wrap_routing_error("todo", account_obj.get("provider", "todo"), "write_failed")

    except Exception as e:
        logger.error(f"Failed to delete tasks: {e}", exc_info=True)
        return wrap_routing_error("todo", target_provider or "local", "write_failed")


# =============================================================================
# set_reminder
# =============================================================================


async def _preview_set_reminder(args: dict, context) -> str:
    card = {
        "card_type": "reminder_draft",
        "title": args.get("reminder_message", ""),
        "when": args.get("human_readable_time") or args.get("schedule_datetime", ""),
        "options": ["approve", "edit", "decline"],
    }
    return "⏰ **Reminder draft**\n\n<!-- inline_card:" + json.dumps(card, ensure_ascii=False) + " -->"


async def _preview_important_date(args: dict, context) -> str:
    card = {
        "card_type": "reminder_draft",
        "title": args.get("title", ""),
        "when": args.get("date", ""),
        "detail": args.get("category", "custom"),
        "options": ["approve", "edit", "decline"],
    }
    return (
        "🎂 **Important date draft**\n\n<!-- inline_card:"
        + json.dumps(card, ensure_ascii=False)
        + " -->"
    )


@tool(needs_approval=True, get_preview=_preview_set_reminder)
async def set_reminder(
    schedule_datetime: Annotated[
        str, "ISO 8601 datetime when the reminder should fire (e.g., 2024-01-15T14:30:00)."
    ],
    reminder_message: Annotated[str, "What to remind the user about."],
    schedule_type: Annotated[str, "Whether this is a one-time or recurring reminder."] = "one_time",
    cron_expression: Annotated[
        Optional[str], "Cron expression for recurring reminders (e.g., '0 8 * * *' for daily 8am)."
    ] = None,
    human_readable_time: Annotated[
        str, "Friendly time description (e.g., 'in 5 minutes', 'every day at 8am')."
    ] = "",
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
        return wrap_routing_error("reminder", "local", "write_failed")

    from koa.triggers.cron.models import (
        AtSchedule,
        CronJobCreate,
        CronScheduleSpec,
        DeliveryConfig,
        DeliveryMode,
        SessionTarget,
        SystemEventPayload,
        WakeMode,
    )

    # Resolve user timezone
    user_tz = context.context_hints.get("timezone", "") if context.context_hints else ""

    # Build schedule
    try:
        if schedule_type == "recurring" and cron_expression:
            schedule = CronScheduleSpec(expr=cron_expression, tz=user_tz or None)
        else:
            if "T" in schedule_datetime:
                local_dt = datetime.fromisoformat(schedule_datetime.replace("Z", "+00:00"))
            else:
                local_dt = datetime.fromisoformat(schedule_datetime)

            # If the datetime has no timezone info, attach the user's timezone
            # so CronService doesn't incorrectly treat it as UTC.
            if local_dt.tzinfo is None and user_tz:
                try:
                    from zoneinfo import ZoneInfo

                    local_dt = local_dt.replace(tzinfo=ZoneInfo(user_tz))
                except Exception:
                    pass

            schedule = AtSchedule(at=local_dt.isoformat())
    except Exception as e:
        logger.error(f"Failed to parse schedule_datetime: {e}")
        return wrap_routing_error("reminder", "local", "write_failed")

    try:
        job = await cron_service.add(
            CronJobCreate(
                name=reminder_message[:50],
                description=f"Reminder: {reminder_message}",
                user_id=context.tenant_id,
                schedule=schedule,
                session_target=SessionTarget.MAIN,
                wake_mode=WakeMode.NOW,
                payload=SystemEventPayload(text=f"Reminder: {reminder_message}"),
                delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                delete_after_run=isinstance(schedule, AtSchedule),
            )
        )

        logger.info(f"Created reminder via CronService: {job.id}")
        time_desc = human_readable_time or schedule_datetime
        return f"Got it! I'll remind you {time_desc}: {reminder_message}"

    except Exception as e:
        logger.error(f"Failed to create reminder: {e}", exc_info=True)
        return wrap_routing_error("reminder", "local", "write_failed")


@tool(needs_approval=True, get_preview=_preview_important_date)
async def remember_important_date(
    title: Annotated[str, "Important date title, e.g. Mom's birthday"],
    date: Annotated[str, "Date like 2026-05-04 or May 4"],
    category: Annotated[str, "birthday, anniversary, holiday, custom"] = "custom",
    notes: Annotated[Optional[str], "Optional note"] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Save a durable important date like a birthday or anniversary to local storage."""
    backend_client = LocalBackendClient.from_context(context)
    normalized_date = _normalize_important_date(date)
    if not normalized_date:
        return "Error: date must be a real date like 2026-05-04 or May 4."

    try:
        result = await backend_client.create_important_date(
            context.tenant_id,
            {
                "title": title,
                "date": normalized_date,
                "category": category,
                "notes": notes,
                "recurring": True,
            },
        )
    except Exception as e:
        logger.error(f"Failed to create important date: {e}", exc_info=True)
        return wrap_routing_error("reminder", "local", "write_failed")

    if result.get("created"):
        return f"Saved {title} for {normalized_date}."
    return wrap_routing_error("reminder", "local", "write_failed")


# =============================================================================
# manage_reminders
# =============================================================================


@tool
async def manage_reminders(
    action: Annotated[str, "What to do with the reminder/automation."],
    task_hint: Annotated[
        str, "Keywords to identify which reminder/automation (for show/update/pause/resume/delete)."
    ] = "",
    status_filter: Annotated[
        str, "Filter by status when listing: 'all', 'active', 'paused' (default 'all')."
    ] = "all",
    new_schedule_datetime: Annotated[
        Optional[str], "New ISO 8601 datetime for update action."
    ] = None,
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
        user_tz = context.context_hints.get("timezone", "") if context.context_hints else ""
        return await _update_reminder(
            cron_service,
            context.tenant_id,
            task_hint,
            new_schedule_datetime,
            new_cron_expression,
            new_message,
            human_readable_time,
            update_type,
            user_tz,
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
    from koa.triggers.cron.models import AtSchedule, CronScheduleSpec, EverySchedule

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
    from koa.triggers.cron.models import CronJobPatch

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
    from koa.triggers.cron.models import CronJobPatch

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
    cron_service,
    tenant_id: str,
    task_hint: str,
    new_schedule_datetime,
    new_cron_expression,
    new_message,
    human_readable_time,
    update_type,
    user_tz: str = "",
) -> str:
    """Update a reminder's schedule or message via CronService."""
    from koa.triggers.cron.models import (
        AgentTurnPayload,
        AtSchedule,
        CronJobPatch,
        CronScheduleSpec,
        SystemEventPayload,
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
                patch.schedule = CronScheduleSpec(expr=new_cron_expression, tz=user_tz or None)
                patch.delete_after_run = False
                time_desc = human_readable_time or _format_cron_expr(new_cron_expression)
                response_parts.append(f"changed to {time_desc}")
            elif new_schedule_datetime:
                try:
                    if "T" in new_schedule_datetime:
                        local_dt = datetime.fromisoformat(
                            new_schedule_datetime.replace("Z", "+00:00")
                        )
                    else:
                        local_dt = datetime.fromisoformat(new_schedule_datetime)
                    patch.schedule = AtSchedule(at=local_dt.isoformat())
                    patch.delete_after_run = True
                    time_desc = human_readable_time or _format_datetime_display(
                        local_dt.isoformat()
                    )
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
# check_overdue_tasks
# =============================================================================


@tool
async def check_overdue_tasks(*, context: AgentToolContext) -> str:
    """Check for overdue and today-due tasks. Used by proactive reminders."""
    try:
        provider, account, error = await _resolve_todo_provider(context, operation="read")
        if error:
            return error

        result = await provider.list_tasks(completed=False)
        if not result.get("success"):
            return wrap_routing_error("todo", account.get("provider", "todo"), "read_failed")

        all_tasks = result.get("data", [])

        if not all_tasks:
            return "No overdue or due tasks. You're all caught up!"

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        overdue = []
        due_today = []

        for task in all_tasks:
            due = task.get("due") or task.get("due_date", "")
            if isinstance(due, dict):
                due = due.get("date", "")
            if not due:
                continue
            due_date = due[:10]  # YYYY-MM-DD
            title = task.get("title", "Untitled")
            if due_date < today_str:
                overdue.append(title)
            elif due_date == today_str:
                due_today.append(title)

        if not overdue and not due_today:
            return "No overdue or due tasks. You're all caught up!"

        parts = []
        if overdue:
            titles = ", ".join(overdue[:5])
            suffix = f" (+{len(overdue) - 5} more)" if len(overdue) > 5 else ""
            parts.append(f"🔴 {len(overdue)} overdue: {titles}{suffix}")
        if due_today:
            titles = ", ".join(due_today[:5])
            suffix = f" (+{len(due_today) - 5} more)" if len(due_today) > 5 else ""
            parts.append(f"📋 {len(due_today)} due today: {titles}{suffix}")

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"check_overdue_tasks failed: {e}", exc_info=True)
        return "Sorry, I couldn't check your tasks right now."


@tool
async def query_local_reminders(
    only_open: Annotated[
        bool, "If True, exclude reminders already marked completed. Default True."
    ] = True,
    max_results: Annotated[int, "Max reminders to return. Default 20."] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Read items from the user's **local iOS Reminders** (synced via the
    Koi app).  Complements cloud todo providers by covering Reminders lists
    that only exist on-device."""
    db = context.context_hints.get("db") if context.context_hints else None
    if db is None:
        return "Local reminders are not available right now."

    from .reminders_sync import pull_reminders

    try:
        rows = await pull_reminders(db, context.tenant_id)
    except Exception as e:
        logger.warning("query_local_reminders failed: %s", e)
        return "Couldn't read local reminders right now."

    if only_open:
        rows = [r for r in rows if not r.get("completed_at")]
    rows = rows[: max(1, int(max_results))]

    if not rows:
        return "No open reminders on this device."

    lines = []
    for r in rows:
        title = r.get("title") or "(Untitled)"
        due = r.get("due_at")
        due_str = f" — due {due.isoformat()}" if due else ""
        lst = r.get("list_name") or ""
        tag = f" [{lst}]" if lst else ""
        lines.append(f"📝 {title}{tag}{due_str}")
    return "\n".join(lines)
