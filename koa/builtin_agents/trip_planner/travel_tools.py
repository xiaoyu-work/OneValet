"""
Travel Tools — Standalone API functions for TravelAgent's mini ReAct loop.

Uses Jina Reader to fetch real-time data from Google Flights and Google Hotels.
Weather data via WeatherAPI.
"""

import logging
import os
import urllib.parse
from typing import Annotated

import httpx

from koa.builtin_agents.tools.jina_reader import jina_fetch
from koa.models import AgentToolContext
from koa.tool_decorator import tool

logger = logging.getLogger(__name__)


# =============================================================================
# search_flights  (via Jina Reader + Google Flights)
# =============================================================================


def _build_google_flights_query_url(
    origin: str, destination: str, date: str, return_date: str = ""
) -> str:
    """Build a Google Flights search URL using natural language query."""
    query = f"flights from {origin} to {destination} on {date}"
    if return_date:
        query += f" return {return_date}"
    return f"https://www.google.com/travel/flights?q={urllib.parse.quote(query)}"


@tool
async def search_flights(
    origin: Annotated[str, "Origin city or airport code"],
    destination: Annotated[str, "Destination city or airport code"],
    date: Annotated[str, "Departure date YYYY-MM-DD"],
    return_date: Annotated[str, "Return date YYYY-MM-DD"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Find real-time flight options with prices from Google Flights."""

    if not origin or not destination or not date:
        return "Error: origin, destination, and date are all required."

    url = _build_google_flights_query_url(origin, destination, date, return_date)
    logger.info(
        "Flight search via Jina Reader: %s -> %s on %s | URL: %s", origin, destination, date, url
    )

    content = await jina_fetch(url, max_chars=20000)
    if not content:
        return (
            f"Could not retrieve flight data for {origin} → {destination} on {date}. "
            f"You can search manually: {url}"
        )

    # Extract just the flight results section to save tokens
    lines = content.split("\n")
    result_lines = []
    in_results = False
    for line in lines:
        stripped = line.strip()
        if any(
            kw in stripped.lower()
            for kw in ["departing flight", "top departing", "search results", "sorted by"]
        ):
            in_results = True
        if in_results:
            result_lines.append(line)
        # Also capture price/cheapest indicators before results section
        if not in_results and ("$" in stripped or "round trip" in stripped.lower()):
            if any(c.isdigit() for c in stripped):
                result_lines.append(line)

    if result_lines:
        extracted = "\n".join(result_lines)
    else:
        extracted = content

    # Truncate to reasonable size for LLM
    if len(extracted) > 8000:
        extracted = extracted[:8000] + "\n\n[Results truncated]"

    return (
        f"Google Flights results for {origin} → {destination} on {date}"
        + (f" (return {return_date})" if return_date else "")
        + f"\nSource: {url}\n\n{extracted}"
    )


# =============================================================================
# search_hotels  (via Jina Reader + Google Hotels)
# =============================================================================


def _build_google_hotels_url(location: str, check_in: str, check_out: str = "") -> str:
    """Build a Google Hotels search URL."""
    query = f"hotels in {location}"
    if check_in:
        query += f" {check_in}"
    if check_out:
        query += f" to {check_out}"
    return f"https://www.google.com/travel/hotels?q={urllib.parse.quote(query)}"


@tool
async def search_hotels(
    location: Annotated[str, "Destination city"],
    check_in: Annotated[str, "Check-in YYYY-MM-DD"],
    check_out: Annotated[str, "Check-out YYYY-MM-DD"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Find real-time hotel options with prices from Google Hotels."""

    if not location or not check_in:
        return "Error: location and check_in date are required."

    url = _build_google_hotels_url(location, check_in, check_out)
    logger.info(
        "Hotel search via Jina Reader: %s (%s to %s) | URL: %s", location, check_in, check_out, url
    )

    content = await jina_fetch(url, max_chars=20000)
    if not content:
        return (
            f"Could not retrieve hotel data for {location} on {check_in}. "
            f"You can search manually: {url}"
        )

    # Truncate to reasonable size for LLM
    if len(content) > 8000:
        content = content[:8000] + "\n\n[Results truncated]"

    return (
        f"Google Hotels results for {location} ({check_in}"
        + (f" to {check_out}" if check_out else "")
        + f")\nSource: {url}\n\n{content}"
    )


# =============================================================================
# get_weather  (unchanged — WeatherAPI works fine)
# =============================================================================


@tool
async def get_weather(
    location: Annotated[str, "City name"],
    days: Annotated[int, "Offset days from today (0..14)"] = 0,
    *,
    context: AgentToolContext,
) -> str:
    """Get current or forecast weather for a city."""

    if not location:
        return "Error: location is required."

    api_key = os.getenv("WEATHER_API_KEY", "")
    if not api_key:
        return "Weather API key not configured."

    try:
        days = max(0, min(14, int(days)))
    except (ValueError, TypeError):
        days = 0

    try:
        async with httpx.AsyncClient() as client:
            if days == 0:
                url = "http://api.weatherapi.com/v1/current.json"
                params = {"key": api_key, "q": location, "aqi": "no"}
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                current = data["current"]
                loc = data["location"]
                return (
                    f"Current weather in {loc['name']}, {loc['country']}:\n"
                    f"- Temperature: {int(current['temp_f'])}°F ({int(current['temp_c'])}°C)\n"
                    f"- Condition: {current['condition']['text']}\n"
                    f"- Feels like: {int(current['feelslike_f'])}°F\n"
                    f"- Humidity: {current['humidity']}%\n"
                    f"- Wind: {int(current['wind_mph'])} mph"
                )
            else:
                url = "http://api.weatherapi.com/v1/forecast.json"
                params = {"key": api_key, "q": location, "days": str(days + 1), "aqi": "no"}
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                loc = data["location"]
                forecast_days = data["forecast"]["forecastday"]
                day_data = forecast_days[min(days, len(forecast_days) - 1)]["day"]

                return (
                    f"Weather forecast for {loc['name']}, {loc['country']} "
                    f"(+{days} day{'s' if days > 1 else ''}):\n"
                    f"- High: {int(day_data['maxtemp_f'])}°F ({int(day_data['maxtemp_c'])}°C)\n"
                    f"- Low: {int(day_data['mintemp_f'])}°F ({int(day_data['mintemp_c'])}°C)\n"
                    f"- Condition: {day_data['condition']['text']}\n"
                    f"- Humidity: {day_data['avghumidity']}%\n"
                    f"- Chance of rain: {day_data.get('daily_chance_of_rain', 0)}%"
                )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            return f"Couldn't find {location}. Please check the city name."
        elif e.response.status_code == 401:
            return "Weather service authentication failed."
        return f"Couldn't get weather for {location}. Try again later?"
    except Exception as e:
        logger.error(f"Weather API failed: {e}", exc_info=True)
        return "Couldn't get the weather. Try again later?"
