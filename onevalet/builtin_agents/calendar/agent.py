"""
CalendarAgent - Agent for all calendar-related requests.

Replaces the separate CalendarAgent, CreateEventAgent, UpdateEventAgent, and
DeleteEventAgent with a single agent that has its own mini ReAct loop.
The orchestrator sees only one "CalendarAgent" tool instead of four separate ones.

The internal LLM decides which tools to call (query_events, create_event,
update_event, delete_event) based on the user's request.
"""

from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent

from .tools import (
    query_events,
    create_event,
    update_event,
    delete_event,
)


@valet(capabilities=["calendar"])
class CalendarAgent(StandardAgent):
    """Check schedule, create, update, or delete calendar events. Use when the user asks about their schedule, meetings, appointments, or wants to create/change/cancel an event."""

    max_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a calendar management assistant with access to the user's calendar.

Available tools:
- query_events: Search and list calendar events by time range or keywords.
- create_event: Create a new calendar event (requires title and start time).
- update_event: Update an existing event (reschedule, rename, change location).
- delete_event: Delete calendar events matching search criteria.

Today's date: {today} ({weekday})

Instructions:
1. If the user's request is missing critical information (event title, time), \
ASK the user for it in your text response WITHOUT calling any tools.
2. Once you have enough information, call the relevant tool.
3. For queries like "what's on my calendar today", call query_events with time_range="today".
4. For creating events, extract the title, start time, and any other details from the user's message.
5. For updating events, identify the target event and the requested changes.
6. For deleting events, identify the events to remove by title or time range.
7. After getting tool results, present the information clearly to the user."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
        )

    tools = [query_events, create_event, update_event, delete_event]
