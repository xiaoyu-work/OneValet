"""CronAgent — agent for creating, listing, updating, and managing cron jobs.

Follows the TodoAgent pattern: a single StandardAgent with a mini ReAct loop
that decides which cron tools to call based on the user's request.
"""

from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent

from .tools import (
    cron_status,
    cron_list,
    cron_add,
    cron_update,
    cron_remove,
    cron_run,
    cron_runs,
)


@valet(capabilities=["cron", "schedule", "timer", "automation", "recurring"])
class CronAgent(StandardAgent):
    """Create, list, update, and manage scheduled cron jobs and recurring automations. Use when the user wants to schedule recurring tasks, set up timed automations, create reminders, or manage existing scheduled jobs."""

    max_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a cron job management assistant with access to scheduling tools.

Available tools:
- cron_status: Show overall cron system status (job counts, next run).
- cron_list: List all cron jobs for the user.
- cron_add: Create a new cron job with a schedule and instruction.
- cron_update: Update an existing cron job (rename, reschedule, enable/disable).
- cron_remove: Delete a cron job permanently.
- cron_run: Manually trigger a cron job to run immediately.
- cron_runs: View the run history for a specific cron job.

Today's date: {today} ({weekday}), timezone: {timezone}

Schedule types:
- "at": One-shot at a specific datetime (ISO 8601). Example: "2025-12-25T08:00:00"
- "every": Recurring interval in seconds. Example: "3600" = every hour, "300" = every 5 min
- "cron": Cron expression (5 fields). Examples: "0 8 * * *" = daily 8am, "*/5 * * * *" = every 5 min, "0 9 * * 1-5" = weekdays 9am

Session targets:
- "isolated": Fresh context each run (default, best for recurring tasks)
- "main": Runs with conversation history (for context-aware tasks)

Instructions:
1. For creating schedules, determine the right schedule_type and schedule_value from the user's request.
2. Convert natural language times to cron expressions or ISO datetimes.
3. Always confirm what you created with the user.
4. For managing jobs (list, update, remove), use the job name or ID.
5. When the user says "every morning at 8am", use cron "0 8 * * *".
6. When the user says "in 30 minutes", calculate the ISO datetime and use "at".
7. When the user says "every 5 minutes", use "every" with value "300"."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        try:
            tz_name = now.astimezone().tzinfo.tzname(now) or "UTC"
        except Exception:
            tz_name = "UTC"
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
            timezone=tz_name,
        )

    tools = (cron_status, cron_list, cron_add, cron_update, cron_remove, cron_run, cron_runs)
