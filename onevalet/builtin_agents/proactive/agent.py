"""
ProactiveCheckAgent — runs periodic checks and notifies the user.

Designed to be triggered by CronService. Calls calendar, task, and
subscription check tools, then formats a concise notification.
"""

from datetime import datetime, timezone

from onevalet import valet
from onevalet.standard_agent import StandardAgent


@valet(domain="productivity")
class ProactiveCheckAgent(StandardAgent):
    """Run proactive checks for upcoming events, overdue tasks, and expiring subscriptions.
    Used by the cron system for periodic notifications."""

    max_turns = 3

    _SYSTEM_PROMPT_TEMPLATE = """\
You are Koi's proactive notification system. Your job is to check for things
the user needs to know RIGHT NOW and format a brief, actionable notification.

Today: {today} ({weekday}), time: {current_time}, timezone: {timezone}

Rules:
1. Check what's relevant based on the instruction.
2. Only include items that need IMMEDIATE attention.
3. If nothing needs attention, respond with EXACTLY: "nothing_to_report"
   (The delivery system will skip the notification.)
4. Keep notifications SHORT — 2-3 lines max. No headers or formatting.
5. Be warm but concise: "Your standup starts in 15 min 📅" not "You have an upcoming event..."
"""

    def get_system_prompt(self) -> str:
        now, tz_name = self._user_now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
            current_time=now.strftime("%H:%M"),
            timezone=tz_name,
        )

    tools = ()  # Tools will be provided by the orchestrator based on user's connected services
