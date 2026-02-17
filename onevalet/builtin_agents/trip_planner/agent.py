"""
TripPlannerAgent - Cross-domain travel planning agent.

This agent orchestrates multiple domains to produce executable itineraries:
- travel market data (flights, hotels, weather)
- local city exploration and routing (places, directions)
- optional execution (calendar events, task creation) after user approval
"""

from datetime import datetime

from onevalet import InputField, valet
from onevalet.standard_agent import StandardAgent, AgentTool

from .travel_tools import (
    search_flights,
    search_hotels,
    check_weather,
)
from onevalet.builtin_agents.maps.tools import (
    search_places,
    get_directions,
)
from onevalet.builtin_agents.calendar.tools import (
    query_events,
    create_event,
)
from onevalet.builtin_agents.todo.tools import (
    create_task,
)


async def _preview_create_event(args: dict, context) -> str:
    summary = args.get("summary", "Untitled event")
    start = args.get("start", "")
    end = args.get("end", "")
    location = args.get("location", "")
    lines = ["Add this event to calendar?"]
    lines.append(f"Title: {summary}")
    if start:
        lines.append(f"Start: {start}")
    if end:
        lines.append(f"End: {end}")
    if location:
        lines.append(f"Location: {location}")
    return "\n".join(lines)


async def _preview_create_task(args: dict, context) -> str:
    title = args.get("title", "Untitled task")
    due = args.get("due", "")
    priority = args.get("priority", "")
    lines = ["Create this trip task?"]
    lines.append(f"Task: {title}")
    if due:
        lines.append(f"Due: {due}")
    if priority:
        lines.append(f"Priority: {priority}")
    return "\n".join(lines)


@valet(capabilities=["travel_planning", "travel", "maps", "itinerary"])
class TripPlannerAgent(StandardAgent):
    """Plan a complete trip itinerary with day-by-day schedule. Use when the user asks to plan a trip, make an itinerary, or organize a multi-day travel plan. Coordinates flights, hotels, weather, places, directions, and optionally creates calendar events and tasks."""

    # Only destination is truly required. Everything else can be inferred
    # by the ReAct LLM (dates from "三天", origin from user profile, etc.).
    # Pattern: "Assume and proceed" — state assumptions, let user correct.
    destination = InputField(
        prompt="Which city or destination are you traveling to?",
        description="Trip destination city/area",
    )

    max_domain_turns = 8

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a senior trip planner that builds realistic, executable itineraries.

Today's date: {today} ({weekday})

## Handling Missing Info
Use what the user provides. For what's missing:
- **Dates**: infer from duration (e.g. "三天" = 3 days starting {tomorrow}). This is the ONLY thing you may infer.
- **Origin city**: if not given, skip flight search entirely. Do NOT guess a city.
- **Budget / preferences**: do not mention or guess. Just plan a balanced trip.
- **Do NOT list assumptions.** Never fabricate information the user didn't provide.
Proceed directly with tool calls. Do NOT ask clarifying questions.

## Tool Usage (CRITICAL)
You MUST call tools to gather real data before producing any plan.
Never generate a plan from your training data alone.

On your FIRST turn, call these tools in parallel:
1. check_weather — destination weather forecast
2. search_places — attractions, restaurants, points of interest
3. search_hotels — accommodation options
4. search_flights — ONLY if the user provided an origin city. Otherwise skip.

You may also use:
- get_directions — verify travel times between locations
- query_events — check calendar for conflicts
- create_event / create_task — only after explicit user approval

Do NOT produce a text-only answer without calling tools first.

## Plan Format — USE TOOL DATA
Your itinerary MUST reference the actual data returned by tools. Include:
- **Weather**: temperature, conditions, what to wear
- **Places**: name, address, rating, opening hours, estimated visit time
- **Hotels**: name, price per night, location
- **Flights**: airline, departure/arrival times, price

Structure:
- **Weather & Clothing** (from check_weather)
- **Day 1..N** with morning / afternoon / evening blocks — each POI with address, rating, and hours
- **Accommodation Options** (from search_hotels, if available)
- **Flight Options** (from search_flights, only if origin was provided)
- **Estimated Budget**

Keep routing realistic: avoid long zig-zag travel within a day.
Only execute write actions (calendar/todo) after explicit user consent.

## Formatting Rules
- Use compact Markdown. NO consecutive blank lines — one blank line max between sections.
- Use the user's language (Chinese if the user writes in Chinese).
- Keep the response concise but information-dense. Avoid filler text.
"""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        from datetime import timedelta
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
            tomorrow=tomorrow,
        )

    async def on_running(self, msg):
        return await super().on_running(msg)

    domain_tools = [
        AgentTool(
            name="check_weather",
            description="Get current or forecast weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Offset days from today (0..14)"},
                },
                "required": ["location"],
            },
            executor=check_weather,
        ),
        AgentTool(
            name="search_places",
            description="Find restaurants, attractions, and points of interest for a location.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "location": {"type": "string", "description": "City or area"},
                },
                "required": ["query"],
            },
            executor=search_places,
        ),
        AgentTool(
            name="get_directions",
            description="Get route distance and duration between two locations.",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Start location"},
                    "destination": {"type": "string", "description": "End location"},
                    "mode": {
                        "type": "string",
                        "enum": ["driving", "walking", "bicycling", "transit"],
                        "description": "Transit mode",
                    },
                },
                "required": ["origin", "destination"],
            },
            executor=get_directions,
        ),
        AgentTool(
            name="search_flights",
            description="Find flight options with prices and schedules.",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin city or IATA code"},
                    "destination": {"type": "string", "description": "Destination city or IATA code"},
                    "date": {"type": "string", "description": "Departure date YYYY-MM-DD"},
                    "return_date": {"type": "string", "description": "Return date YYYY-MM-DD"},
                },
                "required": ["origin", "destination", "date"],
            },
            executor=search_flights,
        ),
        AgentTool(
            name="search_hotels",
            description="Find hotel options with nightly prices.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Destination city"},
                    "check_in": {"type": "string", "description": "Check-in YYYY-MM-DD"},
                    "check_out": {"type": "string", "description": "Check-out YYYY-MM-DD"},
                },
                "required": ["location", "check_in"],
            },
            executor=search_hotels,
        ),
        AgentTool(
            name="query_events",
            description="Check existing calendar events for conflicts.",
            parameters={
                "type": "object",
                "properties": {
                    "time_range": {"type": "string", "description": "Time range, e.g. this week"},
                    "query": {"type": "string", "description": "Optional event keyword"},
                    "max_results": {"type": "integer", "description": "Max events"},
                },
                "required": ["time_range"],
            },
            executor=query_events,
        ),
        AgentTool(
            name="create_event",
            description="Create a calendar event for itinerary booking or schedule lock-in.",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start": {"type": "string", "description": "Event start datetime/natural language"},
                    "end": {"type": "string", "description": "Event end datetime/natural language"},
                    "description": {"type": "string", "description": "Event details"},
                    "location": {"type": "string", "description": "Event location"},
                    "attendees": {"type": "string", "description": "Comma-separated attendee emails"},
                },
                "required": ["summary", "start"],
            },
            executor=create_event,
            needs_approval=True,
            get_preview=_preview_create_event,
        ),
        AgentTool(
            name="create_task",
            description="Create a pre-trip task item (packing, booking, visa, etc.).",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "due": {"type": "string", "description": "Due date YYYY-MM-DD"},
                    "priority": {"type": "string", "description": "low/medium/high/urgent"},
                    "account": {"type": "string", "description": "Todo account name"},
                },
                "required": ["title"],
            },
            executor=create_task,
            needs_approval=True,
            get_preview=_preview_create_task,
        ),
    ]


