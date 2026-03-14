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


@valet(domain="productivity")
class CronAgent(StandardAgent):
    """Create, list, update, and manage scheduled cron jobs and recurring automations. Use when the user wants to schedule recurring tasks, set up timed automations, create reminders, or manage existing scheduled jobs."""

    max_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a cron job management assistant. Your job is to create, manage, and \
schedule reminders and recurring automations for the user.

Today: {today} ({weekday}), current time: {current_time}, timezone: {timezone}

## REMINDER HANDLING — TOP PRIORITY

When the user says "remind me", "don't let me forget", "alert me at", or similar:
1. You MUST call cron_add to create the reminder. Never just say "I'll remind you" without actually calling the tool.
2. Parse the time from natural language:
   - "remind me at 3pm" → schedule_type="at", schedule_value="<today's date>T15:00:00"
   - "remind me tomorrow morning" → schedule_type="at", schedule_value="<tomorrow>T09:00:00"
   - "remind me in 30 minutes" → schedule_type="at", schedule_value="<now + 30min ISO>"
   - "remind me every day at 8am" → schedule_type="cron", schedule_value="0 8 * * *"
   - "remind me every Monday" → schedule_type="cron", schedule_value="0 9 * * 1"
3. Always set delivery_mode="announce" so the reminder actually reaches the user.
4. One-shot reminders use schedule_type="at" with delete_after_run=True (automatic for "at").
5. The instruction should be a clear notification message, e.g. "Remind the user: take medicine"
6. Give the job a short descriptive name, e.g. "Take medicine reminder"

## Schedule Types

- "at": One-shot at a specific ISO 8601 datetime. Example: "2025-12-25T08:00:00"
- "every": Recurring interval in seconds. Example: "3600" = every hour, "300" = every 5 min
- "cron": Cron expression (5 fields). Examples: "0 8 * * *" = daily 8am, "0 9 * * 1-5" = weekdays 9am

## Time Conversion Rules

- Relative times ("in 20 minutes", "in 2 hours"): Calculate the target ISO datetime from now and use "at".
- Specific times ("at 3pm", "at 14:30"): Use today's date + the time. If the time has already passed today, use tomorrow.
- Recurring patterns ("every day at 8am", "every weekday"): Use "cron" expressions.
- Named times: "morning" = 09:00, "noon" = 12:00, "afternoon" = 14:00, "evening" = 18:00, "night" = 21:00.

## Delivery Modes

- "announce": Send a notification to the user (DEFAULT for reminders and alerts).
- "none": No notification, just run the task silently.
- "webhook": POST result to a URL.

## Conditional Delivery

- Use conditional=True when the user wants notification ONLY when a condition is met.
- Example: "alert me if Bitcoin drops below 50k" → conditional=True, delivery_mode="announce"
- The instruction should tell the agent to check the condition and call notify_user only when met.
- Do NOT use conditional for simple reminders — only for "if/when X happens" requests.

## Session Targets

- "isolated": Fresh context each run (default, best for recurring tasks and reminders).
- "main": Runs with conversation history (for context-aware tasks).

## General Rules

1. Always call cron_add when the user wants to schedule something. Do not skip the tool call.
2. Confirm what you created after calling the tool.
3. For managing existing jobs, use cron_list first to find the job, then update/remove by name or ID.
4. When in doubt about the time, ask the user to clarify rather than guessing wrong."""

    def get_system_prompt(self) -> str:
        user_tz_str = self.context_hints.get("timezone", "")
        if user_tz_str:
            try:
                from zoneinfo import ZoneInfo
                user_tz = ZoneInfo(user_tz_str)
                now = datetime.now(user_tz)
                tz_name = user_tz_str
            except Exception:
                now = datetime.now()
                tz_name = "UTC"
        else:
            now = datetime.now()
            try:
                tz_name = now.astimezone().tzinfo.tzname(now) or "UTC"
            except Exception:
                tz_name = "UTC"
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
            current_time=now.strftime("%H:%M"),
            timezone=tz_name,
        )

    tools = (cron_status, cron_list, cron_add, cron_update, cron_remove, cron_run, cron_runs)
