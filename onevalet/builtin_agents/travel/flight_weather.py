"""
Flight Weather Agent - Check destination weather for today's flights

Queries trips table for today's flights, fetches weather for each destination.
Used in morning digest pipeline.
"""
import logging
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet
class FlightWeatherAgent(StandardAgent):
    """Gets today's flights from trips table and fetches destination weather"""

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

    def needs_approval(self) -> bool:
        return False

    def _get_db_client(self):
        """Get database client from context_hints"""
        return self.context_hints.get("db_client")

    async def on_running(self, msg: Message) -> AgentResult:
        """Get today's flights from trips and fetch destination weather"""
        try:
            db_client = self._get_db_client()
            if not db_client:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=""
                )

            trips = db_client.get_today_trips(self.tenant_id)
            flights = [t for t in trips if t.get("trip_type") == "flight"]

            if not flights:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=""
                )

            results = []
            seen_cities = set()

            for flight in flights:
                dest_city = flight.get("destination_city") or flight.get("destination_code") or ""
                if not dest_city or dest_city.lower() in seen_cities:
                    continue

                seen_cities.add(dest_city.lower())
                flight_number = flight.get("trip_number", "")

                logger.info(f"Fetching weather for: {dest_city}")

                weather = await self._get_weather(dest_city)

                if weather:
                    if flight_number:
                        results.append(f"{flight_number} to {dest_city} - {weather}")
                    else:
                        results.append(f"flight to {dest_city} - {weather}")
                else:
                    if flight_number:
                        results.append(f"{flight_number} to {dest_city}")

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="\n".join(results) if results else ""
            )

        except Exception as e:
            logger.error(f"FlightWeatherAgent failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=""
            )

    async def _get_weather(self, city: str) -> str:
        """Fetch weather for a city. Returns short summary or None on error."""
        try:
            from .weather import WeatherAgent

            weather_agent = WeatherAgent(
                tenant_id=self.tenant_id,
                llm_client=self.llm_client
            )

            result = await weather_agent.reply(
                Message(name="user", content=f"weather in {city}", role="user")
            )

            if result.status == AgentStatus.COMPLETED and result.raw_message:
                return await self._summarize_weather(city, result.raw_message)

            return None

        except Exception as e:
            logger.error(f"Weather fetch failed for {city}: {e}")
            return None

    async def _summarize_weather(self, city: str, weather_data: str) -> str:
        """Summarize weather in one short line"""
        if not self.llm_client:
            return weather_data[:100]

        prompt = f"""Summarize this weather for {city} in ONE short line (under 50 chars).
Include: temperature and condition only.

Weather data:
{weather_data}

Example outputs:
- "72F sunny"
- "45F rainy, bring umbrella"
- "28F snow expected"

Your summary (one line, under 50 chars):"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Summarize weather briefly."},
                    {"role": "user", "content": prompt}
                ],
                enable_thinking=False
            )

            return result.content.strip()

        except Exception as e:
            logger.error(f"Weather summary failed: {e}")
            lines = weather_data.split('\n')
            return lines[0][:50] if lines else ""
