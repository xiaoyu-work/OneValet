"""
TravelAgent - Domain agent for all travel-related requests.

Replaces the separate FlightSearchAgent, HotelSearchAgent, and WeatherAgent
with a single agent that has its own mini ReAct loop. The orchestrator sees
only one "TravelAgent" tool instead of three separate ones.

The internal LLM decides which tools to call (search_flights, search_hotels,
check_weather) based on the user's request, and asks clarifying questions
if information is missing (e.g., departure city, travel dates).
"""

from datetime import datetime

from onevalet import valet
from onevalet.agents.domain_agent import DomainAgent, DomainTool

from .tools import search_flights, search_hotels, check_weather


@valet(capabilities=["travel"])
class TravelAgent(DomainAgent):
    """Search real-time flights, hotels, and weather for travel. Use when the user asks about flights, hotels, accommodation, airfare, weather at a destination, or any single travel query."""

    max_domain_turns = 6

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a travel planning assistant with access to real-time search tools.

Available tools:
- search_flights: Search real-time flight offers. Needs origin, destination, date.
- search_hotels: Search real-time hotel offers. Needs location, check_in date.
- check_weather: Check current or forecast weather. Needs location.

Today's date: {today} ({weekday})

Instructions:
1. If the user's request is missing critical information (departure city, destination, travel dates), \
ASK the user for it in your text response WITHOUT calling any tools.
2. Once you have enough information, call the relevant tools.
3. For a full travel plan (e.g., "plan a trip to X"), call ALL relevant tools: \
search_flights, search_hotels, AND check_weather.
4. After getting tool results, synthesize a comprehensive response for the user.
5. Be helpful and proactive — suggest travel tips based on the weather and destination."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )

    domain_tools = [
        DomainTool(
            name="search_flights",
            description="Search real-time flight offers. Returns available flights with prices, airlines, and schedules.",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Departure city name or IATA airport code (e.g., 'Seattle' or 'SEA')",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination city name or IATA airport code (e.g., 'Tokyo' or 'NRT')",
                    },
                    "date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD format",
                    },
                    "return_date": {
                        "type": "string",
                        "description": "Return date in YYYY-MM-DD format (optional, omit for one-way)",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
            executor=search_flights,
        ),
        DomainTool(
            name="search_hotels",
            description="Search real-time hotel offers. Returns available hotels with prices and ratings.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or area name to search hotels in",
                    },
                    "check_in": {
                        "type": "string",
                        "description": "Check-in date in YYYY-MM-DD format",
                    },
                    "check_out": {
                        "type": "string",
                        "description": "Check-out date in YYYY-MM-DD format (optional, defaults to 1 night)",
                    },
                },
                "required": ["location", "check_in"],
            },
            executor=search_hotels,
        ),
        DomainTool(
            name="check_weather",
            description="Check current weather or forecast (up to 14 days ahead) for a location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or location name",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Days from today (0 = current weather, 1 = tomorrow, max 14). Default 0.",
                    },
                },
                "required": ["location"],
            },
            executor=check_weather,
        ),
    ]
