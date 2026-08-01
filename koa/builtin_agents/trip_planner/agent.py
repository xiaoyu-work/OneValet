"""
TripPlannerAgent - Cross-domain travel planning agent.

This agent orchestrates multiple domains to produce executable itineraries:
- travel market data (flights, hotels, weather)
- local city exploration and routing (places, directions)
- optional execution (calendar events, task creation) after user approval
"""

from koa import InputField, valet
from koa.builtin_agents.calendar.tools import create_event, query_events
from koa.builtin_agents.maps.tools import get_directions, search_places
from koa.builtin_agents.todo.tools import create_task
from koa.standard_agent import StandardAgent

from .travel_tools import get_weather, search_flights, search_hotels


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

    max_turns = 20

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
1. get_weather — destination weather forecast
2. search_places — attractions, restaurants, points of interest
3. search_hotels — accommodation options (returns real Google Hotels data with prices)
4. search_flights — ONLY if the user provided an origin city. Returns real Google Flights data with prices and airlines.

You may also use:
- get_directions — verify travel times between locations. If it returns an error, skip it and estimate travel times yourself. Do NOT retry more than once.
- query_events — check calendar for conflicts
- create_event / create_task — only after explicit user approval

Do NOT produce a text-only answer without calling tools first.
If a tool call fails or returns an error, do NOT retry more than once — proceed with the data you have and note what was unavailable.

## IMPORTANT: Use ONLY real data from tools
- Flight prices, airlines, and schedules come from Google Flights via search_flights. Present them exactly as returned.
- Hotel names, prices, and ratings come from Google Hotels via search_hotels. Present them exactly as returned.
- Do NOT fabricate or estimate prices. If a tool returns no data, say so honestly.
- Include the Google Flights / Google Hotels source URL so the user can book directly.

## Plan Format — USE TOOL DATA
Your itinerary MUST reference the actual data returned by tools. Include:
- **Weather**: temperature, conditions, what to wear
- **Places**: name, address, rating, opening hours, estimated visit time
- **Hotels**: name, price per night, location — from search_hotels results
- **Flights**: airline, departure/arrival times, price — from search_flights results, with source link

Structure:
- **Weather & Clothing** (from get_weather)
- **Day 1..N** with morning / afternoon / evening blocks — each POI with address, rating, and hours
- **Accommodation Options** (from search_hotels, if available)
- **Flight Options** (from search_flights, only if origin was provided) with Google Flights link
- **Estimated Budget** based on actual tool data

Keep routing realistic: avoid long zig-zag travel within a day.
Only execute write actions (calendar/todo) after explicit user consent.

## Formatting Rules
- Use compact Markdown. NO consecutive blank lines — one blank line max between sections.
- Use the user's language (Chinese if the user writes in Chinese).
- Keep the response concise but information-dense. Avoid filler text.
"""

    def get_system_prompt(self) -> str:
        now, _ = self._user_now()
        from datetime import timedelta

        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        prompt = self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime("%Y-%m-%d"),
            weekday=now.strftime("%A"),
            tomorrow=tomorrow,
        )

        # Inject loyalty programs / travel preferences from user profile
        profile = self.context_hints.get("user_profile", {}) if self.context_hints else {}
        travel = profile.get("travel", {})
        loyalty = travel.get("loyalty_programs", [])
        if loyalty:
            lines = ["\n## User Travel Preferences"]
            for prog in loyalty:
                name = prog.get("program", "")
                status = prog.get("status", "")
                prog_type = prog.get("type", "")
                if name:
                    label = f"- {name}"
                    if status:
                        label += f" ({status})"
                    if prog_type == "airline":
                        label += " — prefer this airline when available"
                    elif prog_type == "hotel":
                        label += " — prefer this hotel chain when available"
                    lines.append(label)
            prompt += "\n".join(lines)

        return prompt

    async def on_running(self, msg):
        return await super().on_running(msg)

    tools = (
        get_weather,
        search_places,
        get_directions,
        search_flights,
        search_hotels,
        query_events,
        create_event,
        create_task,
    )
