"""
CalendarDomainAgent - Domain agent for all calendar-related requests.

Replaces the separate CalendarAgent, CreateEventAgent, UpdateEventAgent, and
DeleteEventAgent with a single agent that has its own mini ReAct loop.
The orchestrator sees only one "CalendarDomainAgent" tool instead of four separate ones.

The internal LLM decides which tools to call (query_events, create_event,
update_event, delete_event) based on the user's request.
"""

from datetime import datetime

from onevalet import valet
from onevalet.agents.domain_agent import DomainAgent, DomainTool

from .tools import (
    query_events,
    create_event,
    _preview_create_event,
    update_event,
    _preview_update_event,
    delete_event,
    _preview_delete_event,
)


@valet(capabilities=["calendar"])
class CalendarDomainAgent(DomainAgent):
    """Check schedule, create, update, or delete calendar events. Use when the user asks about their schedule, meetings, appointments, or wants to create/change/cancel an event."""

    max_domain_turns = 5

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

    domain_tools = [
        DomainTool(
            name="query_events",
            description="Search and list calendar events. Returns events matching the time range and optional keyword query.",
            parameters={
                "type": "object",
                "properties": {
                    "time_range": {
                        "type": "string",
                        "description": "Time range to search (e.g., 'today', 'tomorrow', 'this week', 'next week', 'this month', 'next 3 days')",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional keywords to search in event titles",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of events to return (default 10)",
                    },
                },
                "required": ["time_range"],
            },
            executor=query_events,
        ),
        DomainTool(
            name="create_event",
            description="Create a new calendar event. Requires at least a title and start time.",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Event title/summary",
                    },
                    "start": {
                        "type": "string",
                        "description": "Event start time (e.g., 'tomorrow at 2pm', '2025-03-15 14:00')",
                    },
                    "end": {
                        "type": "string",
                        "description": "Event end time (optional, defaults to 1 hour after start)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description/details (optional)",
                    },
                    "location": {
                        "type": "string",
                        "description": "Event location (optional)",
                    },
                    "attendees": {
                        "type": "string",
                        "description": "Comma-separated list of attendee email addresses (optional)",
                    },
                },
                "required": ["summary", "start"],
            },
            executor=create_event,
            needs_approval=True,
            get_preview=_preview_create_event,
        ),
        DomainTool(
            name="update_event",
            description="Update an existing calendar event. Specify the target event and what to change.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Keywords to identify the event (title, person's name, time reference like 'my 2pm meeting')",
                    },
                    "changes": {
                        "type": "object",
                        "description": "What to change",
                        "properties": {
                            "new_time": {
                                "type": "string",
                                "description": "New start time if rescheduling",
                            },
                            "new_title": {
                                "type": "string",
                                "description": "New event title if renaming",
                            },
                            "new_location": {
                                "type": "string",
                                "description": "New location if changing location",
                            },
                            "new_duration": {
                                "type": "string",
                                "description": "New duration if changing length (e.g., '2 hours')",
                            },
                        },
                    },
                },
                "required": ["target", "changes"],
            },
            executor=update_event,
            needs_approval=True,
            get_preview=_preview_update_event,
        ),
        DomainTool(
            name="delete_event",
            description="Delete calendar events matching the search criteria.",
            parameters={
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Keywords to search for events to delete (event title, keywords)",
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range to search (e.g., 'today', 'tomorrow', 'this week'). Defaults to 'next 7 days'.",
                    },
                },
                "required": ["search_query"],
            },
            executor=delete_event,
            needs_approval=True,
            get_preview=_preview_delete_event,
        ),
    ]
