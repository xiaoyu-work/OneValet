"""
TripPlannerAgent - Cross-domain travel planning agent.

This agent orchestrates multiple domains to produce executable itineraries:
- travel market data (flights, hotels, weather)
- local city exploration and routing (places, directions)
- optional execution (calendar events, task creation) after user approval
"""

from datetime import datetime

from onevalet import InputField, valet
from onevalet.standard_agent import StandardAgent

from .travel_tools import search_flights, search_hotels, check_weather
from onevalet.builtin_agents.maps.tools import search_places, get_directions
from onevalet.builtin_agents.calendar.tools import query_events, create_event
from onevalet.builtin_agents.todo.tools import create_task


@valet(domain="travel")
class TripPlannerAgent(StandardAgent):
    """Plan a complete trip itinerary with day-by-day schedule. Use when the user asks to plan a trip, make an itinerary, or organize a multi-day travel plan. Coordinates flights, hotels, weather, places, directions, and optionally creates calendar events and tasks."""

    # Only destination is truly required. Everything else can be inferred
    # by the ReAct LLM (dates from "3 days", origin from user profile, etc.).
    # Pattern: "Assume and proceed" — state assumptions, let user correct.
    destination = InputField(
        prompt="Which city or destination are you traveling to?",
        description="Trip destination city/area",
    )

    max_turns = 8

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a senior trip planner that builds realistic, executable itineraries.

Today's date: {today} ({weekday})

## Handling Missing Info
Use what the user provides. For what's missing:
- **Dates**: infer from duration (e.g. "3 days" = 3 days starting {tomorrow}). This is the ONLY thing you may infer.
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

    tools = (
        check_weather,
        search_places,
        get_directions,
        search_flights,
        search_hotels,
        query_events,
        create_event,
        create_task,
    )


