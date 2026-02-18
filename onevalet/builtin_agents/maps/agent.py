"""
MapsAgent - Agent for all maps and location-related requests.

Replaces the separate MapSearchAgent, DirectionsAgent, and AirQualityAgent
with a single agent that has its own mini ReAct loop. The orchestrator sees
only one "MapsAgent" tool instead of three separate ones.

The internal LLM decides which tools to call (search_places, get_directions,
check_air_quality) based on the user's request.
"""

from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent

from .tools import search_places, get_directions, check_air_quality


@valet(capabilities=["maps"])
class MapsAgent(StandardAgent):
    """Find places, restaurants, attractions, get directions, and check air quality. Use when the user asks about nearby places, how to get somewhere, navigation, or local recommendations."""

    max_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a maps and location assistant with access to real-time search tools.

Available tools:
- search_places: Search for places, restaurants, businesses by query and location.
- get_directions: Get driving/transit/walking directions between two locations.
- check_air_quality: Check current air quality (AQI) for a location.

Today's date: {today} ({weekday})

Instructions:
1. If the user's request is missing critical information (location, destination, origin), \
ASK the user for it in your text response WITHOUT calling any tools.
2. Once you have enough information, call the relevant tools.
3. After getting tool results, synthesize a clear, helpful response for the user.
4. For directions, if the user says "from home" and you don't have their address, ask them.
5. Be helpful and proactive — suggest nearby alternatives or additional info when relevant."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )

    tools = (
        search_places,
        get_directions,
        check_air_quality,
    )
