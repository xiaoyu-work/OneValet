"""Cron Tools — CRUD operations for cron job management.

Matches OpenClaw's cron-tool.ts actions:
status, list, add, update, remove, run, runs.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Optional

from onevalet.models import AgentToolContext
from onevalet.tool_decorator import tool

logger = logging.getLogger(__name__)


def _get_service(context: AgentToolContext):
    """Get CronService from context hints."""
    if context.context_hints:
        return context.context_hints.get("cron_service")
    return None


def _format_ms(ms: Optional[int]) -> str:
    """Format millisecond timestamp to readable string."""
    if ms is None:
        return "—"
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_relative(ms: Optional[int]) -> str:
    """Format ms timestamp as relative time from now."""
    if ms is None:
        return "—"
    import time
    now = int(time.time() * 1000)
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


def _format_schedule(job) -> str:
    """Format a job's schedule for display."""
    s = job.schedule
    kind = getattr(s, "kind", "?")
    if kind == "at":
        return f"once at {s.at}"
    elif kind == "every":
        secs = s.every_ms / 1000
        if secs >= 3600:
            return f"every {secs / 3600:.1f}h"
        elif secs >= 60:
            return f"every {secs / 60:.0f}m"
        return f"every {secs:.0f}s"
    elif kind == "cron":
        tz_str = f" ({s.tz})" if s.tz else ""
        return f"cron: {s.expr}{tz_str}"
    return kind


# =============================================================================
# cron_status
# =============================================================================

@tool
async def cron_status(*, context: AgentToolContext) -> str:
    """Show overall cron system status including job counts and next scheduled run."""
    service = _get_service(context)
    if not service:
        return "Cron service is not available."

    status = await service.status()
    lines = [
        f"Cron scheduler: {'running' if status['running'] else 'stopped'}",
        f"Total jobs: {status['total_jobs']}",
        f"Enabled: {status['enabled_jobs']}",
        f"Currently running: {status['running_jobs']}",
    ]
    if status["next_due_at_ms"]:
        lines.append(f"Next run: {_format_ms(status['next_due_at_ms'])} ({_format_relative(status['next_due_at_ms'])})")
    else:
        lines.append("Next run: none scheduled")
    return "\n".join(lines)


# =============================================================================
# cron_list
# =============================================================================

@tool
async def cron_list(
    include_disabled: Annotated[bool, "Whether to include disabled jobs"] = False,
    *, context: AgentToolContext,
) -> str:
    """List all cron jobs for the current user."""
    service = _get_service(context)
    if not service:
        return "Cron service is not available."

    jobs = service.list_jobs(user_id=context.tenant_id, include_disabled=include_disabled)
    if not jobs:
        return "No cron jobs found."

    lines = []
    for job in jobs:
        status_icon = "✓" if job.enabled else "✗"
        last = ""
        if job.state.last_run_status:
            last = f" | last: {job.state.last_run_status}"
            if job.state.last_run_status == "error" and job.state.last_error:
                last += f" ({job.state.last_error[:50]})"
        next_str = _format_relative(job.state.next_run_at_ms) if job.enabled else "disabled"

        lines.append(
            f"[{status_icon}] {job.name} (id: {job.id[:8]}…)\n"
            f"    Schedule: {_format_schedule(job)} | Target: {job.session_target.value}\n"
            f"    Next: {next_str}{last}"
        )

    return "\n\n".join(lines)


# =============================================================================
# cron_add
# =============================================================================

@tool
async def cron_add(
    name: Annotated[str, "Name for the cron job"],
    instruction: Annotated[str, "What the agent should do when the job fires"],
    schedule_type: Annotated[str, "Schedule type: 'at' (one-shot datetime), 'every' (recurring interval), 'cron' (cron expression)"],
    schedule_value: Annotated[str, "Schedule value: ISO datetime for 'at', seconds number for 'every', cron expression for 'cron' (e.g. '0 8 * * *')"],
    timezone: Annotated[str, "IANA timezone for cron expressions (e.g. 'America/Los_Angeles')"] = "",
    session_target: Annotated[str, "Execution mode: 'main' (with context) or 'isolated' (fresh)"] = "isolated",
    delivery_mode: Annotated[str, "How to deliver results: 'none', 'announce' (notify user), or 'webhook'"] = "none",
    delivery_channel: Annotated[Optional[str], "Channel for announce delivery"] = None,
    webhook_url: Annotated[Optional[str], "URL for webhook delivery"] = None,
    conditional: Annotated[bool, "If true, only notify when a condition is met. The agent will have a notify_user tool to decide when to send notifications. Use for 'alert me when X happens' type requests."] = False,
    delete_after_run: Annotated[bool, "Auto-delete after successful execution (default true for one-shot)"] = False,
    *, context: AgentToolContext,
) -> str:
    """Create a new cron job with the specified schedule and configuration."""
    service = _get_service(context)
    if not service:
        return "Cron service is not available."

    from onevalet.triggers.cron.models import (
        AtSchedule, EverySchedule, CronScheduleSpec,
        SessionTarget, WakeMode,
        SystemEventPayload, AgentTurnPayload,
        DeliveryMode, DeliveryConfig,
        CronJobCreate,
    )

    # Build schedule
    if schedule_type == "at":
        schedule = AtSchedule(at=schedule_value)
    elif schedule_type == "every":
        try:
            seconds = float(schedule_value)
            schedule = EverySchedule(every_ms=int(seconds * 1000))
        except ValueError:
            return f"Invalid interval value: {schedule_value}. Provide seconds (e.g. '3600' for 1 hour)."
    elif schedule_type == "cron":
        schedule = CronScheduleSpec(
            expr=schedule_value,
            tz=timezone or None,
            stagger_ms=5 * 60 * 1000 if not timezone else None,  # 5 min default stagger
        )
    else:
        return f"Unknown schedule type: {schedule_type}. Use 'at', 'every', or 'cron'."

    # Build payload based on session target
    target = SessionTarget(session_target)
    if target == SessionTarget.MAIN:
        payload = SystemEventPayload(text=instruction)
    else:
        payload = AgentTurnPayload(message=instruction)

    # Build delivery
    delivery = None
    if delivery_mode != "none" or conditional:
        mode = DeliveryMode(delivery_mode) if delivery_mode != "none" else DeliveryMode.ANNOUNCE
        delivery = DeliveryConfig(
            mode=mode,
            channel=delivery_channel,
            webhook_url=webhook_url,
            conditional=conditional,
        )

    input_data = CronJobCreate(
        name=name,
        user_id=context.tenant_id,
        schedule=schedule,
        session_target=target,
        wake_mode=WakeMode.NOW if target == SessionTarget.MAIN else WakeMode.NEXT_HEARTBEAT,
        payload=payload,
        delivery=delivery,
        delete_after_run=delete_after_run if schedule_type != "at" else True,
    )

    try:
        job = await service.add(input_data)
        next_str = _format_relative(job.state.next_run_at_ms)
        return (
            f"Created cron job: {job.name}\n"
            f"ID: {job.id}\n"
            f"Schedule: {_format_schedule(job)}\n"
            f"Target: {job.session_target.value}\n"
            f"Next run: {next_str}"
        )
    except Exception as e:
        return f"Failed to create cron job: {e}"


# =============================================================================
# cron_update
# =============================================================================

@tool
async def cron_update(
    job_hint: Annotated[str, "Name or ID of the cron job to update"],
    enabled: Annotated[Optional[bool], "Enable or disable the job"] = None,
    new_name: Annotated[Optional[str], "New name for the job"] = None,
    new_instruction: Annotated[Optional[str], "New instruction/message for the job"] = None,
    new_schedule_type: Annotated[Optional[str], "New schedule type: 'at', 'every', or 'cron'"] = None,
    new_schedule_value: Annotated[Optional[str], "New schedule value"] = None,
    new_timezone: Annotated[Optional[str], "New timezone for cron expressions"] = None,
    *, context: AgentToolContext,
) -> str:
    """Update an existing cron job's configuration."""
    service = _get_service(context)
    if not service:
        return "Cron service is not available."

    job = service.find_job(job_hint, user_id=context.tenant_id)
    if not job:
        return f"No cron job found matching '{job_hint}'."

    from onevalet.triggers.cron.models import (
        AtSchedule, EverySchedule, CronScheduleSpec,
        SystemEventPayload, AgentTurnPayload,
        CronJobPatch,
    )

    patch = CronJobPatch()

    if enabled is not None:
        patch.enabled = enabled
    if new_name is not None:
        patch.name = new_name

    # Update schedule if specified
    if new_schedule_type and new_schedule_value:
        if new_schedule_type == "at":
            patch.schedule = AtSchedule(at=new_schedule_value)
        elif new_schedule_type == "every":
            try:
                patch.schedule = EverySchedule(every_ms=int(float(new_schedule_value) * 1000))
            except ValueError:
                return f"Invalid interval: {new_schedule_value}"
        elif new_schedule_type == "cron":
            patch.schedule = CronScheduleSpec(expr=new_schedule_value, tz=new_timezone)

    # Update instruction
    if new_instruction:
        if isinstance(job.payload, SystemEventPayload):
            patch.payload = SystemEventPayload(text=new_instruction)
        else:
            patch.payload = AgentTurnPayload(message=new_instruction)

    try:
        updated = await service.update(job.id, patch)
        return (
            f"Updated cron job: {updated.name}\n"
            f"Enabled: {updated.enabled}\n"
            f"Schedule: {_format_schedule(updated)}\n"
            f"Next run: {_format_relative(updated.state.next_run_at_ms)}"
        )
    except Exception as e:
        return f"Failed to update: {e}"


# =============================================================================
# cron_remove
# =============================================================================

@tool
async def cron_remove(
    job_hint: Annotated[str, "Name or ID of the cron job to remove"],
    *, context: AgentToolContext,
) -> str:
    """Delete a cron job permanently."""
    service = _get_service(context)
    if not service:
        return "Cron service is not available."

    job = service.find_job(job_hint, user_id=context.tenant_id)
    if not job:
        return f"No cron job found matching '{job_hint}'."

    removed = await service.remove(job.id)
    if removed:
        return f"Deleted cron job: {job.name} ({job.id})"
    return f"Failed to delete job '{job_hint}'."


# =============================================================================
# cron_run
# =============================================================================

@tool
async def cron_run(
    job_hint: Annotated[str, "Name or ID of the cron job to run immediately"],
    *, context: AgentToolContext,
) -> str:
    """Manually trigger a cron job to run immediately, outside its normal schedule."""
    service = _get_service(context)
    if not service:
        return "Cron service is not available."

    job = service.find_job(job_hint, user_id=context.tenant_id)
    if not job:
        return f"No cron job found matching '{job_hint}'."

    try:
        entry = await service.run(job.id, mode="force")
        lines = [f"Ran cron job: {job.name}"]
        lines.append(f"Status: {entry.status or 'unknown'}")
        if entry.summary:
            lines.append(f"Result: {entry.summary[:500]}")
        if entry.error:
            lines.append(f"Error: {entry.error}")
        if entry.duration_ms:
            lines.append(f"Duration: {entry.duration_ms}ms")
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to run job: {e}"


# =============================================================================
# cron_runs
# =============================================================================

@tool
async def cron_runs(
    job_hint: Annotated[str, "Name or ID of the cron job to view history for"],
    limit: Annotated[int, "Maximum number of runs to show"] = 10,
    *, context: AgentToolContext,
) -> str:
    """View the run history for a specific cron job."""
    service = _get_service(context)
    if not service:
        return "Cron service is not available."

    job = service.find_job(job_hint, user_id=context.tenant_id)
    if not job:
        return f"No cron job found matching '{job_hint}'."

    runs = await service.get_runs(job.id, limit=limit)
    if not runs:
        return f"No run history for '{job.name}'."

    lines = [f"Run history for '{job.name}' (last {len(runs)}):"]
    for entry in runs:
        time_str = _format_ms(entry.ts)
        duration = f" ({entry.duration_ms}ms)" if entry.duration_ms else ""
        status = entry.status or "?"
        summary = ""
        if entry.summary:
            summary = f" — {entry.summary[:100]}"
        if entry.error:
            summary = f" — Error: {entry.error[:100]}"
        delivery = ""
        if entry.delivered is not None:
            delivery = f" [{'delivered' if entry.delivered else 'not delivered'}]"
        lines.append(f"  {time_str} | {status}{duration}{delivery}{summary}")

    return "\n".join(lines)
