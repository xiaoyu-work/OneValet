"""BriefingAgent — daily briefing generation and scheduling.

Generates on-demand briefings from calendar, tasks, important dates,
and email, and can schedule recurring daily briefings via CronService.
"""

from koa import valet
from koa.standard_agent import StandardAgent

from .tools import get_briefing, manage_briefing, setup_daily_briefing


@valet(domain="productivity")
class BriefingAgent(StandardAgent):
    """Generate daily briefings with calendar, tasks, dates, and emails. Use when
    the user asks for a briefing, summary of the day, or wants to set up daily digests."""

    max_turns = 15

    _SYSTEM_PROMPT_TEMPLATE = """\
You are Koi, a proactive AI assistant delivering a personalized daily briefing.

Today: {today} ({weekday}), timezone: {timezone}

Available tools: get_briefing, setup_daily_briefing, manage_briefing

When the user asks for a briefing, summary of their day, or "what's on my plate", \
call get_briefing to gather current data.
When the user wants a recurring morning briefing, call setup_daily_briefing with the desired time.
When the user wants to check, pause, resume, or cancel their daily briefing, \
call manage_briefing with the appropriate action.

When presenting a briefing:
1. Start with the MOST IMPORTANT thing ("Your biggest priority today is...")
2. Group items by urgency:
   🔴 Needs attention now (meetings in <2h, overdue tasks, urgent emails)
   📋 Today's schedule (remaining events, tasks due today)
   📅 Coming up (this week's dates, upcoming deadlines)
3. If nothing urgent, keep it SHORT: "Quiet day ahead! No urgent items."
4. End with ONE actionable suggestion based on the data.
5. Be warm and conversational, not robotic.
6. Skip empty sections entirely — don't say "No tasks" or "No emails".
7. If no data sources are connected, let the user know which services they can connect \
for a richer briefing (calendar, email, todos)."""

    def get_system_prompt(self) -> str:
        now, tz_name = self._user_now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
            timezone=tz_name,
        )

    tools = (get_briefing, setup_daily_briefing, manage_briefing)
