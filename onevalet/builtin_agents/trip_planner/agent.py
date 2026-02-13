"""
TripPlannerAgent - Cross-domain travel planning agent.

This agent orchestrates multiple domains to produce executable itineraries:
- travel market data (flights, hotels, weather)
- local city exploration and routing (places, directions)
- optional execution (calendar events, task creation) after user approval
"""

from datetime import datetime
import re
from typing import Any, Dict

from onevalet import InputField, valet
from onevalet.agents.domain_agent import DomainAgent, DomainTool
from onevalet.result import AgentStatus

from onevalet.builtin_agents.travel.tools import (
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


def _validate_date(value: str):
    if not value:
        return "Date is required in YYYY-MM-DD format."
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return None
    except ValueError:
        return "Use YYYY-MM-DD format (example: 2026-03-15)."


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
class TripPlannerAgent(DomainAgent):
    """Plan a complete trip itinerary with day-by-day schedule. Use when the user asks to plan a trip, make an itinerary, or organize a multi-day travel plan. Coordinates flights, hotels, weather, places, directions, and optionally creates calendar events and tasks."""

    destination = InputField(
        prompt="Which city or destination are you traveling to?",
        description="Trip destination city/area",
    )
    start_date = InputField(
        prompt="What is your trip start date? (YYYY-MM-DD)",
        description="Trip start date in YYYY-MM-DD",
        validator=_validate_date,
    )
    end_date = InputField(
        prompt="What is your trip end date? (YYYY-MM-DD)",
        description="Trip end date in YYYY-MM-DD",
        validator=_validate_date,
    )
    origin = InputField(
        prompt="What city are you departing from?",
        description="Origin city for flights",
        required=False,
    )
    budget = InputField(
        prompt="What is your total budget range for this trip?",
        description="Budget range for planning",
        required=False,
    )
    preferences = InputField(
        prompt="Any preferences? (food, pace, neighborhoods, kid-friendly, nightlife, museums)",
        description="Traveler preferences for itinerary",
        required=False,
    )

    max_domain_turns = 8

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a senior trip planner that builds realistic, executable itineraries.

You can use tools from travel, maps, calendar, and todo domains.

Today's date: {today} ({weekday})

Workflow requirements:
1. For itinerary requests, gather evidence first. Do not produce a full plan before at least:
   - weather for destination
   - place suggestions for the destination
2. If user provides origin and dates, also gather flights and hotels.
3. If key data is missing, ask concise clarifying questions.
4. Build day-by-day schedules with morning/afternoon/evening blocks.
5. Keep routing realistic: avoid long zig-zag travel within a day.
6. Mark assumptions explicitly when user did not provide details.
7. Only execute write actions (calendar/todo) after explicit user consent.

Output requirements:
- Include "Plan Summary", "Day 1..N", "Estimated Budget", and "Assumptions".
- Include references to the tool findings you used.
"""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
        )

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Deterministic extraction to avoid LLM hallucinating required fields."""
        text = (user_input or "").strip()
        if not text:
            return {}

        extracted: Dict[str, Any] = {}

        # Explicit ISO dates only. Do not infer from vague phrases like "three days".
        dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if len(dates) >= 2:
            extracted["start_date"] = dates[0]
            extracted["end_date"] = dates[1]
        elif len(dates) == 1:
            if "start" in text.lower() or "from" in text.lower() or "开始" in text or "出发" in text:
                extracted["start_date"] = dates[0]
            elif "end" in text.lower() or "to" in text.lower() or "结束" in text or "返回" in text:
                extracted["end_date"] = dates[0]

        # Basic destination hints (Chinese/English travel phrases)
        dest_patterns = [
            r"去\s*([\u4e00-\u9fa5A-Za-z\s\-]+?)(?:\d|天|日|旅游|旅行|行程|$)",
            r"到\s*([\u4e00-\u9fa5A-Za-z\s\-]+?)(?:\d|天|日|旅游|旅行|行程|$)",
            r"to\s+([A-Za-z\s\-]+?)(?:\s+for\s+\d|\s+trip|\s+itinerary|$)",
        ]
        for pattern in dest_patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                city = m.group(1).strip(" ,，。")
                if city:
                    extracted["destination"] = city
                    break

        origin_match = re.search(r"from\s+([A-Za-z\s\-]+?)(?:\s+to|$)", text, flags=re.IGNORECASE)
        if origin_match:
            extracted["origin"] = origin_match.group(1).strip(" ,")
        else:
            zh_origin = re.search(r"从\s*([\u4e00-\u9fa5A-Za-z\s\-]+?)\s*(出发|到|去)", text)
            if zh_origin:
                extracted["origin"] = zh_origin.group(1).strip(" ,，。")

        if re.search(r"\$|预算|budget|人民币|美元|CNY|USD", text, flags=re.IGNORECASE):
            extracted["budget"] = text

        if re.search(r"(喜欢|偏好|avoid|prefer|museum|nightlife|food|亲子|慢节奏)", text, flags=re.IGNORECASE):
            extracted["preferences"] = text

        # Fallback for destination only: allow superclass LLM extraction for city names,
        # but never trust it for required date fields.
        if "destination" not in extracted:
            llm_extracted = await super().extract_fields(user_input)
            if isinstance(llm_extracted, dict):
                if llm_extracted.get("destination"):
                    extracted["destination"] = llm_extracted.get("destination")
                if llm_extracted.get("origin") and "origin" not in extracted:
                    extracted["origin"] = llm_extracted.get("origin")

        return extracted

    async def on_running(self, msg):
        # Hard gate: never proceed if required fields are missing.
        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt() or "Please provide the missing trip details.",
                missing_fields=missing,
                metadata={
                    "requires_user_input": True,
                    "missing_fields": missing,
                },
            )

        # Cross-field date consistency check
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            if end < start:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message="Your end date is earlier than start date. Please provide valid dates in YYYY-MM-DD.",
                    missing_fields=["start_date", "end_date"],
                    metadata={
                        "requires_user_input": True,
                        "missing_fields": ["start_date", "end_date"],
                    },
                )
        except Exception:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="Please provide valid start and end dates in YYYY-MM-DD format.",
                missing_fields=["start_date", "end_date"],
                metadata={
                    "requires_user_input": True,
                    "missing_fields": ["start_date", "end_date"],
                },
            )

        return await super().on_running(msg)

    domain_tools = [
        DomainTool(
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
        DomainTool(
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
        DomainTool(
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
        DomainTool(
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
        DomainTool(
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
        DomainTool(
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
        DomainTool(
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
        DomainTool(
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

