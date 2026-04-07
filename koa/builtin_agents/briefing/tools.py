"""Briefing Tools — daily briefing generation and scheduling.

Provides on-demand briefing generation, daily briefing cron setup,
and management of the scheduled briefing job.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated

from koa.models import AgentToolContext
from koa.tool_decorator import tool

logger = logging.getLogger(__name__)

BRIEFING_JOB_NAME = "Daily Briefing"
BRIEFING_INSTRUCTION = (
    "Generate my morning briefing. Call get_briefing to gather today's data, "
    "then present the results following these rules:\n"
    "1. Lead with the MOST urgent item (overdue task, meeting in <2h, important date today)\n"
    "2. Use this format — skip sections with nothing to report:\n"
    "   🔴 Urgent: [items needing immediate action]\n"
    "   📅 Today: [calendar events with times]\n"
    "   ✅ Tasks: [due today or overdue]\n"
    "   🎂 Dates: [birthdays/anniversaries within 7 days]\n"
    "   📧 Emails: [unread important emails]\n"
    "3. End with ONE actionable suggestion\n"
    "4. Max 8 lines total. Be warm but concise."
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


def _is_within_hours(iso_str: str, hours: int) -> bool:
    """Return True if *iso_str* is an ISO-8601 datetime within *hours* from now."""
    if not iso_str:
        return False
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return now <= dt <= now + timedelta(hours=hours)
    except (ValueError, TypeError):
        return False


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
# Profile birthday helper
# =============================================================================

def _check_profile_birthdays(profile: dict, now: datetime, sections: list):
    """Check user profile relationships for upcoming birthdays within 7 days."""
    relationships = profile.get("relationships", {})
    family = relationships.get("family", [])
    partner = relationships.get("significant_other")
    # Collect all people with birthdays
    people = list(family)
    if partner and isinstance(partner, dict):
        people.append(partner)

    if not people:
        return

    today = now.date()
    lines = []
    for person in people:
        bday_str = person.get("birthday", "")
        name = person.get("name", "Someone")
        rel = person.get("relationship", "")
        if not bday_str:
            continue
        try:
            bday = datetime.strptime(bday_str, "%Y-%m-%d").date()
            # This year's occurrence
            this_year = bday.replace(year=today.year)
            if this_year < today:
                this_year = bday.replace(year=today.year + 1)
            days_until = (this_year - today).days
            if days_until <= 7:
                label = f"{name}" + (f" ({rel})" if rel else "")
                if days_until == 0:
                    lines.append(f"- 🔴 🎂 {label}'s birthday is TODAY!")
                elif days_until == 1:
                    lines.append(f"- 🎂 {label}'s birthday is tomorrow")
                else:
                    lines.append(f"- 🎂 {label}'s birthday in {days_until} days")
        except (ValueError, TypeError):
            continue

    if lines:
        # Merge with existing dates section or create new
        header_exists = any("## Upcoming Dates" in s for s in sections)
        if header_exists:
            for i, s in enumerate(sections):
                if "## Upcoming Dates" in s:
                    sections[i] = s + "\n" + "\n".join(lines)
                    break
        else:
            sections.append("## Upcoming Dates\n" + "\n".join(lines))


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
                    start_dt = e.get("start", {}).get("dateTime", "")
                    label = "🔴 SOON: " if _is_within_hours(start_dt, 2) else ""
                    lines.append(f"- {label}{time_str}: {e.get('summary', 'Untitled')}")
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
                    due = t.get("due_date") or t.get("due", {}).get("date", "")
                    overdue = "🔴 OVERDUE: " if due and due < now.strftime("%Y-%m-%d") else ""
                    lines.append(f"- {overdue}{t.get('title', 'Untitled')}")
                sections.append("\n".join(lines))
        except Exception as exc:
            logger.debug("Briefing: todo section failed: %s", exc)

    # 3. Important dates (from DB + user profile relationships)
    db = context.context_hints.get("db") if context.context_hints else None
    if db:
        try:
            from ..digest.important_dates_repo import ImportantDatesRepository
            repo = ImportantDatesRepository(db)
            dates = await repo.get_important_dates(context.tenant_id, days_ahead=7)
            if dates:
                lines = ["## Upcoming Dates"]
                today_str = now.strftime("%Y-%m-%d")
                for d in dates:
                    date_str = d.get("upcoming_date", d.get("date", ""))
                    today_mark = "🔴 TODAY: " if date_str == today_str else ""
                    lines.append(f"- {today_mark}{d.get('title', '')}: {date_str}")
                sections.append("\n".join(lines))
        except Exception as exc:
            logger.debug("Briefing: important dates section failed: %s", exc)

    # 3b. Profile relationship birthdays (family, partner, etc.)
    user_profile = context.context_hints.get("user_profile") if context.context_hints else None
    if user_profile:
        try:
            _check_profile_birthdays(user_profile, now, sections)
        except Exception as exc:
            logger.debug("Briefing: profile birthdays failed: %s", exc)

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

    # Count urgent items across all sections
    combined = "\n".join(sections)
    urgent_count = sum(1 for line in combined.split("\n") if "🔴" in line)

    if urgent_count == 0 and len(sections) <= 1:
        return "😌 Quiet day ahead — no urgent meetings, tasks, or emails. Enjoy!"

    summary = f"📊 {urgent_count} urgent item{'s' if urgent_count != 1 else ''} today."
    return summary + "\n\n" + "\n\n".join(sections)


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

    from koa.triggers.cron.models import (
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
        from koa.triggers.cron.models import CronJobPatch
        try:
            await cron_service.update(job.id, CronJobPatch(enabled=False))
            return "Daily briefing has been disabled. Use action 'enable' to re-enable it."
        except Exception as e:
            return f"Failed to disable daily briefing: {e}"

    elif action == "enable":
        if job.enabled:
            return "Daily briefing is already enabled."
        from koa.triggers.cron.models import CronJobPatch
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
