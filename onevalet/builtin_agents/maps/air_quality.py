"""
AirQualityAgent - Get air quality information using Google Air Quality API
"""
import os
import logging
import httpx
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class AirQualityAgent(StandardAgent):
    """Air quality query agent using Google Air Quality API"""

    location = InputField(
        prompt="Which location would you like to check air quality for?",
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
        if not self.llm_client:
            return {"location": user_input}

        try:
            prompt = f"""Extract the location from this user message about air quality.

User message: "{user_input}"

Return ONLY the location name (city or place), nothing else.
If no location is mentioned, return "unknown".

Examples:
- "What's the air quality in Seattle?" -> "Seattle"
- "AQI in Beijing" -> "Beijing"
- "Air quality in NYC" -> "New York City"
- "Is the air good today?" -> "unknown"

Location:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract locations from text."},
                    {"role": "user", "content": prompt}
                ],
                enable_thinking=False
            )

            location = result.content.strip()

            if location.lower() == "unknown":
                return {}

            return {"location": location}

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {"location": user_input}

    async def _geocode_location(self, location: str) -> Dict[str, float]:
        """Convert location name to coordinates using Google Geocoding API"""
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "address": location,
                "key": os.getenv("GOOGLE_MAPS_API_KEY", "")
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

            if data["status"] != "OK" or not data.get("results"):
                logger.error(f"Geocoding failed for {location}: {data.get('status')}")
                return {}

            result = data["results"][0]
            coords = result["geometry"]["location"]

            return {
                "lat": coords["lat"],
                "lng": coords["lng"],
                "formatted_address": result["formatted_address"]
            }

        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return {}

    async def on_running(self, msg: Message) -> AgentResult:
        """Get air quality information for location"""
        location = self.collected_fields.get("location", "unknown")

        logger.info(f"Getting air quality for {location}")

        if not os.getenv("GOOGLE_MAPS_API_KEY"):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Google Maps API key not configured. Please contact support."
            )

        try:
            coords = await self._geocode_location(location)

            if not coords or "lat" not in coords:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't find {location}. Please check the location name."
                )

            url = "https://airquality.googleapis.com/v1/currentConditions:lookup"

            headers = {"Content-Type": "application/json"}

            request_body = {
                "location": {
                    "latitude": coords["lat"],
                    "longitude": coords["lng"]
                },
                "extraComputations": [
                    "HEALTH_RECOMMENDATIONS",
                    "DOMINANT_POLLUTANT_CONCENTRATION",
                    "POLLUTANT_CONCENTRATION",
                    "LOCAL_AQI"
                ],
                "languageCode": "en"
            }

            params = {"key": os.getenv("GOOGLE_MAPS_API_KEY", "")}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=request_body,
                    params=params,
                    timeout=15.0
                )
                response.raise_for_status()
                data = response.json()

            indexes = data.get("indexes", [])
            if not indexes:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"No air quality data available for {location}."
                )

            primary_index = indexes[0]
            aqi_value = primary_index.get("aqi")
            category = primary_index.get("category", "Unknown")
            dominant_pollutant = primary_index.get("dominantPollutant", "Unknown")

            health_recommendations = data.get("healthRecommendations", {})
            general_population = health_recommendations.get("generalPopulation", "No recommendations available")

            air_quality_data = {
                "location": coords.get("formatted_address", location),
                "aqi": aqi_value,
                "category": category,
                "dominant_pollutant": dominant_pollutant,
                "health_advice": general_population
            }

            logger.info(f"Air quality fetched for {location}: AQI {aqi_value}, {category}")

            result_lines = [
                f"Air Quality: {air_quality_data['location']}",
                f"AQI: {air_quality_data['aqi']}",
                f"Category: {air_quality_data['category']}",
                f"Dominant Pollutant: {air_quality_data['dominant_pollutant']}",
                f"Health Advice: {air_quality_data['health_advice']}",
            ]

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="\n".join(result_lines)
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Air Quality API HTTP error: {e.response.status_code}")
            if e.response.status_code == 400:
                error_msg = f"Invalid location: {location}. Please try again."
            elif e.response.status_code in [401, 403]:
                error_msg = "Air Quality API authentication failed. Please contact support."
            else:
                error_msg = f"Couldn't get air quality for {location}. Try again later?"

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=error_msg
            )

        except Exception as e:
            logger.error(f"Air quality API call failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't get air quality data. Try again later?"
            )
