"""
Weather Agent - Simple single-step agent for weather queries
Supports both current weather and forecast (up to 14 days).
"""
import os
import logging
import json
import httpx
from datetime import datetime
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class WeatherAgent(StandardAgent):
    """Get current weather or forecast for a location. Use when the user asks about weather, or when planning travel to check destination conditions."""

    location = InputField(
        prompt="Which city would you like to check the weather for?",
        description="City or location name",
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract location and date from user input using LLM."""
        if not self.llm_client:
            return {"location": user_input, "days_from_today": 0}

        try:
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")
            day_of_week = now.strftime("%A")

            prompt = f"""Extract the location and date from this weather query.

IMPORTANT CONTEXT:
- Today is {day_of_week}, {today_str}
- WeatherAPI supports forecasts up to 14 days ahead

User message: "{user_input}"

Extract and return a JSON object with:
1. "location": City or place name. Return "unknown" if not mentioned.
2. "days_from_today": Number of days from today (0 for today, 1 for tomorrow, etc.). Max 14. Default 0.

Examples (assuming today is {day_of_week}, {today_str}):
- "What's the weather in Beijing?" -> {{"location": "Beijing", "days_from_today": 0}}
- "Weather tomorrow in NYC" -> {{"location": "New York City", "days_from_today": 1}}
- "How about this weekend?" -> {{"location": "unknown", "days_from_today": <days to Saturday>}}
- "Weather next Monday in SF" -> {{"location": "San Francisco", "days_from_today": <days to next Monday>}}

Return ONLY the JSON object, nothing else:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract location and date from weather queries. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                tools=None,
                enable_thinking=False
            )

            response_text = result.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            extracted = json.loads(response_text)

            location = extracted.get("location", "unknown")
            days_from_today = extracted.get("days_from_today", 0)

            days_from_today = max(0, min(14, int(days_from_today)))

            if location.lower() == "unknown":
                location = self._get_location_from_profile()
                if location:
                    logger.info(f"Using location from user profile: {location}")
                else:
                    return {}

            logger.info(f"Extracted: location={location}, days_from_today={days_from_today}")

            return {
                "location": location,
                "days_from_today": days_from_today
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return {}
        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

    def _get_location_from_profile(self) -> str:
        """Get user's home city from profile addresses."""
        try:
            profile = self.context_hints.get("user_profile", {})
            if profile:
                addresses = profile.get("addresses", [])
                for addr in addresses:
                    city = addr.get("city")
                    state = addr.get("state")
                    if city:
                        if state:
                            return f"{city}, {state}"
                        return city
        except Exception as e:
            logger.warning(f"Failed to get location from profile: {e}")
        return ""

    def _parse_current_weather(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse current weather response from WeatherAPI."""
        current = data["current"]
        location_data = data["location"]

        return {
            "location": f"{location_data['name']}, {location_data['country']}",
            "temperature": int(current["temp_f"]),
            "temperature_c": int(current["temp_c"]),
            "condition": current["condition"]["text"],
            "humidity": current["humidity"],
            "wind_speed": int(current["wind_mph"]),
            "feels_like": int(current["feelslike_f"])
        }

    def _parse_forecast_weather(self, data: Dict[str, Any], days_from_today: int) -> Dict[str, Any]:
        """Parse forecast weather response from WeatherAPI."""
        location_data = data["location"]
        forecast_days = data["forecast"]["forecastday"]

        if days_from_today < len(forecast_days):
            day_data = forecast_days[days_from_today]["day"]
        else:
            day_data = forecast_days[-1]["day"]

        return {
            "location": f"{location_data['name']}, {location_data['country']}",
            "temperature": int(day_data["avgtemp_f"]),
            "temperature_c": int(day_data["avgtemp_c"]),
            "high": int(day_data["maxtemp_f"]),
            "high_c": int(day_data["maxtemp_c"]),
            "low": int(day_data["mintemp_f"]),
            "low_c": int(day_data["mintemp_c"]),
            "condition": day_data["condition"]["text"],
            "humidity": day_data["avghumidity"],
            "chance_of_rain": day_data.get("daily_chance_of_rain", 0)
        }

    async def on_running(self, msg: Message) -> AgentResult:
        """Get weather information for location"""
        location = self.collected_fields.get("location", "unknown")
        days_from_today = self.collected_fields.get("days_from_today", 0)

        logger.info(f"Getting weather for {location}, days_from_today={days_from_today}")

        try:
            async with httpx.AsyncClient() as client:
                if days_from_today == 0:
                    url = "http://api.weatherapi.com/v1/current.json"
                    params = {
                        "key": os.getenv("WEATHER_API_KEY", ""),
                        "q": location,
                        "aqi": "no"
                    }
                    response = await client.get(url, params=params, timeout=10.0)
                    response.raise_for_status()
                    data = response.json()

                    weather_data = self._parse_current_weather(data)
                else:
                    url = "http://api.weatherapi.com/v1/forecast.json"
                    params = {
                        "key": os.getenv("WEATHER_API_KEY", ""),
                        "q": location,
                        "days": days_from_today + 1,
                        "aqi": "no"
                    }
                    response = await client.get(url, params=params, timeout=10.0)
                    response.raise_for_status()
                    data = response.json()

                    weather_data = self._parse_forecast_weather(data, days_from_today)

            logger.info(f"Weather fetched for {location}: {weather_data.get('temperature')}F, {weather_data.get('condition')}")

            if days_from_today == 0:
                weather_info = f"""- Location: {weather_data.get('location')}
- Temperature: {weather_data.get('temperature')}F ({weather_data.get('temperature_c')}C)
- Condition: {weather_data.get('condition')}
- Feels like: {weather_data.get('feels_like')}F
- Humidity: {weather_data.get('humidity')}%
- Wind: {weather_data.get('wind_speed')} mph"""
            else:
                weather_info = f"""- Location: {weather_data.get('location')}
- High: {weather_data.get('high')}F ({weather_data.get('high_c')}C)
- Low: {weather_data.get('low')}F ({weather_data.get('low_c')}C)
- Condition: {weather_data.get('condition')}
- Humidity: {weather_data.get('humidity')}%
- Chance of rain: {weather_data.get('chance_of_rain')}%"""

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=weather_info
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"WeatherAPI HTTP error: {e.response.status_code}")
            if e.response.status_code == 400:
                error_msg = f"Couldn't find {location}. Please check the city name."
            elif e.response.status_code == 401:
                error_msg = "Weather service authentication failed."
            else:
                error_msg = f"Couldn't get the weather for {location}. Try again later?"

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=error_msg
            )

        except Exception as e:
            logger.error(f"Weather API call failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't get the weather. Try again later?"
            )
