"""
ProactiveCheckAgent — runs periodic checks and notifies the user.

Designed to be triggered by CronService. Calls calendar, task, and
subscription check tools, then formats a concise notification.
"""

from datetime import datetime, timezone

from koa import valet
from koa.standard_agent import StandardAgent
from .habit_discovery import analyze_user_habits


@valet(domain="productivity")
class ProactiveCheckAgent(StandardAgent):
    """Run proactive checks for upcoming events, overdue tasks, and expiring subscriptions.
    Used by the cron system for periodic notifications."""

    max_turns = 3

    _SYSTEM_PROMPT_TEMPLATE = """\
You are Koi's proactive notification system. Your job is to check for things
the user needs to know RIGHT NOW and format a brief, actionable notification.

Today: {today} ({weekday}), time: {current_time}, timezone: {timezone}

Checklist — check ALL of these and only report items that need attention:
- Calendar: any event starting within 2 hours?
- Tasks: any overdue or due today?
- Important dates: any birthday/anniversary within 3 days?
- Packages: any arriving today?
- Subscriptions: any renewing within 3 days?

Rules:
1. Only include items that need IMMEDIATE attention.
2. If nothing needs attention, respond with EXACTLY: "nothing_to_report"
   (The delivery system will skip the notification.)
3. Keep notifications SHORT — 2-3 lines max. Use emoji prefixes.
4. Be warm but concise: "Your standup starts in 15 min 📅" not "You have an upcoming event..."
"""

    def get_system_prompt(self) -> str:
        now, tz_name = self._user_now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
            current_time=now.strftime("%H:%M"),
            timezone=tz_name,
        )

    tools = (analyze_user_habits,)  # Additional tools provided by the orchestrator based on user's connected services
