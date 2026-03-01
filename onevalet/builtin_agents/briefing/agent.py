"""BriefingAgent — daily briefing generation and scheduling.

Generates on-demand briefings from calendar, tasks, important dates,
and email, and can schedule recurring daily briefings via CronService.
"""

from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent

from .tools import get_briefing, setup_daily_briefing, manage_briefing


@valet(domain="productivity")
class BriefingAgent(StandardAgent):
    """Generate daily briefings with calendar, tasks, dates, and emails. Use when
    the user asks for a briefing, summary of the day, or wants to set up daily digests."""

    max_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a daily briefing assistant with access to briefing tools.

Available tools:
- get_briefing: Generate an on-demand briefing with today's calendar events, pending tasks, \
upcoming important dates, and unread emails.
- setup_daily_briefing: Set up or update a recurring daily briefing cron job at a specified time.
- manage_briefing: Manage the daily briefing job (check status, enable, disable, or delete).

Today's date: {today} ({weekday}), timezone: {timezone}

Instructions:
1. When the user asks for a briefing, summary of their day, or "what's on my plate", \
call get_briefing to gather current data.
2. When the user wants a recurring morning briefing ("set up a daily briefing", \
"send me a summary every morning at 8am"), call setup_daily_briefing with the desired time.
3. When the user wants to check, pause, resume, or cancel their daily briefing, \
call manage_briefing with the appropriate action.
4. Present briefing results in a clear, organized format.
5. If no data sources are connected, let the user know which services they can connect \
for a richer briefing (calendar, email, todos).
6. Be concise and helpful. Summarize, don't just dump raw data."""

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

    tools = (get_briefing, setup_daily_briefing, manage_briefing)
