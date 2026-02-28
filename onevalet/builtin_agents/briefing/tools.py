"""Briefing Tools — daily briefing generation and scheduling.

Provides on-demand briefing generation, daily briefing cron setup,
and management of the scheduled briefing job.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated

from onevalet.models import AgentToolContext
from onevalet.tool_decorator import tool

logger = logging.getLogger(__name__)

BRIEFING_JOB_NAME = "Daily Briefing"
BRIEFING_INSTRUCTION = (
    "Generate a morning briefing for the user. Summarize today's calendar events, "
    "pending tasks, upcoming important dates, and unread emails."
)


# =============================================================================
# Helpers
# =============================================================================

def _get_cron_service(context: AgentToolContext):
    """Get CronService from context hints."""
    if context.context_hints:
        return context.context_hints.get("cron_service")
    return None


def _find_briefing_job(cron_service, tenant_id: str):
    """Find the existing Daily Briefing cron job by name, if any."""
    try:
        jobs = cron_service.list_jobs(user_id=tenant_id, include_disabled=True)
    except Exception:
        return None
    for job in jobs:
        if job.name == BRIEFING_JOB_NAME:
            return job
    return None


def _format_schedule(job) -> str:
    """Format a job's schedule for display."""
    s = job.schedule
    kind = getattr(s, "kind", "?")
    if kind == "cron":
        tz_str = f" ({s.tz})" if s.tz else ""
        return f"cron: {s.expr}{tz_str}"
    return kind


def _format_relative(ms) -> str:
    """Format ms timestamp as relative time from now."""
    if ms is None:
        return "not scheduled"
    import time as _time
    now = int(_time.time() * 1000)
    diff_s = (ms - now) / 1000
    if diff_s < 0:
        return f"{abs(diff_s):.0f}s ago"
    if diff_s < 60:
        return f"in {diff_s:.0f}s"
    if diff_s < 3600:
        return f"in {diff_s / 60:.0f}m"
    if diff_s < 86400:
        return f"in {diff_s / 3600:.1f}h"
    return f"in {diff_s / 86400:.1f}d"


# =============================================================================
# get_briefing
# =============================================================================

@tool
async def get_briefing(*, context: AgentToolContext) -> str:
    """Generate a daily briefing with calendar events, pending tasks, important dates, and unread emails."""
    sections = []

    # 1. Calendar events today
    cal_provider = context.context_hints.get("calendar_provider") if context.context_hints else None
    if cal_provider:
        try:
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            today_end = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
            events = await cal_provider.list_events(time_min=today_start, time_max=today_end)
            if events.get("success") and events.get("data"):
                lines = ["## Calendar"]
                for e in events["data"]:
                    time_str = e.get("start", {}).get("dateTime", "All day")
                    lines.append(f"- {time_str}: {e.get('summary', 'Untitled')}")
                sections.append("\n".join(lines))
        except Exception as exc:
            logger.debug("Briefing: calendar section failed: %s", exc)

    # 2. Pending todos
    todo_provider = context.context_hints.get("todo_provider") if context.context_hints else None
    if todo_provider:
        try:
            tasks = await todo_provider.query(status="pending")
            if tasks.get("success") and tasks.get("data"):
                lines = ["## Tasks"]
                for t in tasks["data"][:10]:
                    lines.append(f"- {t.get('title', 'Untitled')}")
                sections.append("\n".join(lines))
        except Exception as exc:
            logger.debug("Briefing: todo section failed: %s", exc)

    # 3. Important dates (from DB)
    db = context.context_hints.get("db") if context.context_hints else None
    if db:
        try:
            from ..digest.important_dates_repo import ImportantDatesRepository
            repo = ImportantDatesRepository(db)
            dates = await repo.get_important_dates(context.tenant_id, days_ahead=7)
            if dates:
                lines = ["## Upcoming Dates"]
                for d in dates:
                    lines.append(f"- {d.get('title', '')}: {d.get('upcoming_date', d.get('date', ''))}")
                sections.append("\n".join(lines))
        except Exception as exc:
            logger.debug("Briefing: important dates section failed: %s", exc)

    # 4. Unread emails
    email_provider = context.context_hints.get("email_provider") if context.context_hints else None
    if email_provider:
        try:
            emails = await email_provider.list_messages(query="is:unread", max_results=5)
            if emails.get("success") and emails.get("data"):
                lines = ["## Unread Emails"]
                for e in emails["data"]:
                    lines.append(f"- {e.get('sender', 'Unknown')}: {e.get('subject', 'No subject')}")
                sections.append("\n".join(lines))
        except Exception as exc:
            logger.debug("Briefing: email section failed: %s", exc)

    if not sections:
        return "No briefing data available. Connect your calendar, email, or todo services first."

    return "\n\n".join(sections)


# =============================================================================
# setup_daily_briefing
# =============================================================================

@tool
async def setup_daily_briefing(
    schedule_time: Annotated[str, "Time for the daily briefing in HH:MM 24-hour format (e.g. '08:00')."] = "08:00",
    tz: Annotated[str, "IANA timezone for scheduling (e.g. 'America/New_York'). Leave empty for server default."] = "",
    *, context: AgentToolContext,
) -> str:
    """Set up or update a recurring daily briefing delivered each morning at the specified time."""
    cron_service = _get_cron_service(context)
    if not cron_service:
        return "Cron service is not available. Cannot schedule daily briefing."

    # Parse time
    try:
        parts = schedule_time.strip().split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("out of range")
    except (ValueError, IndexError):
        return f"Invalid time format: '{schedule_time}'. Please use HH:MM 24-hour format (e.g. '08:00')."

    from onevalet.triggers.cron.models import (
        CronScheduleSpec,
        SessionTarget,
        WakeMode,
        AgentTurnPayload,
        DeliveryConfig,
        DeliveryMode,
        CronJobCreate,
        CronJobPatch,
    )

    cron_expr = f"{minute} {hour} * * *"
    schedule = CronScheduleSpec(
        expr=cron_expr,
        tz=tz or None,
    )

    # Check if a briefing job already exists
    existing = _find_briefing_job(cron_service, context.tenant_id)
    if existing:
        # Update the existing job instead of creating a duplicate
        try:
            patch = CronJobPatch(
                schedule=schedule,
                enabled=True,
            )
            updated = await cron_service.update(existing.id, patch)
            tz_str = f" ({tz})" if tz else ""
            return (
                f"Updated daily briefing schedule to {hour:02d}:{minute:02d}{tz_str}.\n"
                f"Job ID: {updated.id}\n"
                f"Next run: {_format_relative(updated.state.next_run_at_ms)}"
            )
        except Exception as e:
            return f"Failed to update existing daily briefing: {e}"

    # Create new briefing job
    delivery = DeliveryConfig(
        mode=DeliveryMode.ANNOUNCE,
        channel="callback",
    )

    job_input = CronJobCreate(
        name=BRIEFING_JOB_NAME,
        description="Automated daily morning briefing with calendar, tasks, dates, and emails.",
        user_id=context.tenant_id,
        schedule=schedule,
        session_target=SessionTarget.ISOLATED,
        wake_mode=WakeMode.NEXT_HEARTBEAT,
        payload=AgentTurnPayload(message=BRIEFING_INSTRUCTION),
        delivery=delivery,
    )

    try:
        job = await cron_service.add(job_input)
        tz_str = f" ({tz})" if tz else ""
        return (
            f"Daily briefing scheduled at {hour:02d}:{minute:02d}{tz_str}.\n"
            f"Job ID: {job.id}\n"
            f"Schedule: {_format_schedule(job)}\n"
            f"Next run: {_format_relative(job.state.next_run_at_ms)}"
        )
    except Exception as e:
        return f"Failed to create daily briefing: {e}"


# =============================================================================
# manage_briefing
# =============================================================================

@tool
async def manage_briefing(
    action: Annotated[str, "Action to perform on the daily briefing: 'status', 'disable', 'enable', or 'delete'."],
    *, context: AgentToolContext,
) -> str:
    """Manage the daily briefing cron job — check status, enable, disable, or delete it."""
    cron_service = _get_cron_service(context)
    if not cron_service:
        return "Cron service is not available."

    job = _find_briefing_job(cron_service, context.tenant_id)
    if not job:
        return "No daily briefing is currently set up. Use setup_daily_briefing to create one."

    if action == "status":
        status = "enabled" if job.enabled else "disabled"
        last_run = job.state.last_run_status or "never run"
        next_run = _format_relative(job.state.next_run_at_ms) if job.enabled else "disabled"
        lines = [
            f"Daily Briefing: {status}",
            f"Schedule: {_format_schedule(job)}",
            f"Last run: {last_run}",
            f"Next run: {next_run}",
        ]
        if job.state.consecutive_errors:
            lines.append(f"Consecutive errors: {job.state.consecutive_errors}")
        if job.state.last_error:
            lines.append(f"Last error: {job.state.last_error[:100]}")
        return "\n".join(lines)

    elif action == "disable":
        if not job.enabled:
            return "Daily briefing is already disabled."
        from onevalet.triggers.cron.models import CronJobPatch
        try:
            await cron_service.update(job.id, CronJobPatch(enabled=False))
            return "Daily briefing has been disabled. Use action 'enable' to re-enable it."
        except Exception as e:
            return f"Failed to disable daily briefing: {e}"

    elif action == "enable":
        if job.enabled:
            return "Daily briefing is already enabled."
        from onevalet.triggers.cron.models import CronJobPatch
        try:
            updated = await cron_service.update(job.id, CronJobPatch(enabled=True))
            return (
                f"Daily briefing has been re-enabled.\n"
                f"Next run: {_format_relative(updated.state.next_run_at_ms)}"
            )
        except Exception as e:
            return f"Failed to enable daily briefing: {e}"

    elif action == "delete":
        try:
            removed = await cron_service.remove(job.id)
            if removed:
                return f"Daily briefing has been deleted (was job ID: {job.id})."
            return "Failed to delete the daily briefing."
        except Exception as e:
            return f"Failed to delete daily briefing: {e}"

    else:
        return f"Unknown action: '{action}'. Supported actions: status, disable, enable, delete."
