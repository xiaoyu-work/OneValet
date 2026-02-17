"""
Travel Domain Tools — Standalone API functions for TravelAgent's mini ReAct loop.

Extracted from FlightSearchAgent, HotelSearchAgent, and WeatherAgent.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Optional

import httpx

from onevalet.standard_agent import AgentToolContext
from onevalet.tool_decorator import tool

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================

async def _get_amadeus_token() -> Optional[str]:
    """Get Amadeus API OAuth2 access token."""
    api_key = os.getenv("AMADEUS_API_KEY", "")
    api_secret = os.getenv("AMADEUS_API_SECRET", "")
    if not api_key or not api_secret:
        return None

    try:
        url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": api_secret,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        return result.get("access_token")
    except Exception as e:
        logger.error(f"Failed to get Amadeus token: {e}")
        return None


_COMMON_IATA = {
    "seattle": "SEA", "new york": "JFK", "nyc": "JFK",
    "los angeles": "LAX", "la": "LAX", "san francisco": "SFO",
    "chicago": "ORD", "boston": "BOS", "miami": "MIA",
    "atlanta": "ATL", "dallas": "DFW", "denver": "DEN",
    "las vegas": "LAS", "portland": "PDX", "london": "LHR",
    "paris": "CDG", "tokyo": "NRT", "beijing": "PEK",
    "shanghai": "PVG", "hong kong": "HKG", "singapore": "SIN",
    "dubai": "DXB", "sydney": "SYD", "toronto": "YYZ",
}


async def _convert_to_iata(location: str, llm_client: Any) -> str:
    """Convert city name to 3-letter IATA airport code."""
    if not location:
        return ""

    location_lower = location.lower().strip()

    # Already an IATA code
    if len(location) == 3 and location.isalpha():
        return location.upper()

    # Common lookup
    if location_lower in _COMMON_IATA:
        return _COMMON_IATA[location_lower]

    # LLM fallback
    if llm_client:
        try:
            result = await llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You convert locations to IATA airport codes."},
                    {"role": "user", "content": (
                        f'Convert this location to IATA airport code.\n'
                        f'Location: "{location}"\n'
                        f'Return ONLY the 3-letter IATA code, nothing else.\n'
                        f'If you don\'t know, return "UNKNOWN".\nIATA code:'
                    )},
                ],
                enable_thinking=False,
            )
            code = result.content.strip().upper()
            if code != "UNKNOWN" and len(code) == 3:
                return code
        except Exception as e:
            logger.error(f"LLM IATA conversion failed: {e}")

    return location.upper()[:3]


# =============================================================================
# search_flights
# =============================================================================

@tool
async def search_flights(
    origin: Annotated[str, "Origin city or IATA code"],
    destination: Annotated[str, "Destination city or IATA code"],
    date: Annotated[str, "Departure date YYYY-MM-DD"],
    return_date: Annotated[str, "Return date YYYY-MM-DD"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Find flight options with prices and schedules."""

    if not origin or not destination or not date:
        return "Error: origin, destination, and date are all required."

    if not os.getenv("AMADEUS_API_KEY") or not os.getenv("AMADEUS_API_SECRET"):
        return "Flight search unavailable: AMADEUS_API_KEY and AMADEUS_API_SECRET not set. Please configure them in Settings > API Keys."

    token = await _get_amadeus_token()
    if not token:
        return "Couldn't connect to flight search service. Try again later?"

    try:
        origin_code = await _convert_to_iata(origin, context.llm_client)
        dest_code = await _convert_to_iata(destination, context.llm_client)

        logger.info(f"Flight search: {origin}({origin_code}) -> {destination}({dest_code}) on {date}")

        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        params = {
            "originLocationCode": origin_code,
            "destinationLocationCode": dest_code,
            "departureDate": date,
            "adults": "1",
            "max": "5",
            "currencyCode": "USD",
        }
        if return_date:
            params["returnDate"] = return_date

        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers, timeout=20.0)
            response.raise_for_status()
            data = response.json()

        offers = data.get("data", [])
        if not offers:
            return f"No flights found from {origin} to {destination} on {date}. Try different dates?"

        trip_type = f"return {return_date}" if return_date else "one-way"
        lines = [f"Flights {origin_code} → {dest_code} on {date} ({trip_type}):\n"]

        for i, offer in enumerate(offers[:5], 1):
            price = offer.get("price", {}).get("total", "N/A")
            currency = offer.get("price", {}).get("currency", "USD")
            itineraries = offer.get("itineraries", [])
            if not itineraries:
                continue
            outbound = itineraries[0]
            segments = outbound.get("segments", [])
            if not segments:
                continue
            first_seg = segments[0]
            last_seg = segments[-1]
            carrier = first_seg.get("carrierCode", "")
            flight_num = first_seg.get("number", "")
            dep_time = first_seg.get("departure", {}).get("at", "")
            arr_time = last_seg.get("arrival", {}).get("at", "")
            stops = len(segments) - 1
            stops_text = "Direct" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"
            lines.append(f"{i}. {carrier}{flight_num} | {currency} {price} | {stops_text}")
            lines.append(f"   Departs: {dep_time}")
            lines.append(f"   Arrives: {arr_time}")
            if len(itineraries) > 1:
                ret = itineraries[1]
                ret_segs = ret.get("segments", [])
                if ret_segs:
                    r_first = ret_segs[0]
                    r_last = ret_segs[-1]
                    r_stops = len(ret_segs) - 1
                    r_stops_text = "Direct" if r_stops == 0 else f"{r_stops} stop{'s' if r_stops > 1 else ''}"
                    lines.append(
                        f"   Return: {r_first.get('carrierCode', '')}{r_first.get('number', '')} | "
                        f"{r_first.get('departure', {}).get('at', '')} → "
                        f"{r_last.get('arrival', {}).get('at', '')} | {r_stops_text}"
                    )
            lines.append("")

        return "\n".join(lines).strip()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            return "Invalid search: please check your dates and locations."
        elif e.response.status_code == 401:
            return "Flight search authentication failed. Please contact support."
        return "Couldn't search flights. Try again later?"
    except Exception as e:
        logger.error(f"Flight search failed: {e}", exc_info=True)
        return "Couldn't search flights. Try again later?"


# =============================================================================
# search_hotels
# =============================================================================

async def _geocode_location(location: str) -> Optional[Dict[str, float]]:
    """Convert location name to lat/lng via Google Geocoding API."""
    google_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not google_api_key:
        return None

    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": location, "key": google_api_key}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

        if data["status"] != "OK" or not data.get("results"):
            return None

        coords = data["results"][0]["geometry"]["location"]
        return {"latitude": coords["lat"], "longitude": coords["lng"]}
    except Exception as e:
        logger.error(f"Geocoding failed: {e}")
        return None


@tool
async def search_hotels(
    location: Annotated[str, "Destination city"],
    check_in: Annotated[str, "Check-in YYYY-MM-DD"],
    check_out: Annotated[str, "Check-out YYYY-MM-DD"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Find hotel options with nightly prices."""

    if not location or not check_in:
        return "Error: location and check_in date are required."

    # Default to 1 night if check_out not specified
    if not check_out and check_in:
        try:
            check_in_date = datetime.strptime(check_in, "%Y-%m-%d")
            check_out = (check_in_date + timedelta(days=1)).strftime("%Y-%m-%d")
        except ValueError:
            return f"Invalid date format: {check_in}. Use YYYY-MM-DD."

    if not os.getenv("AMADEUS_API_KEY") or not os.getenv("AMADEUS_API_SECRET"):
        return "Hotel search unavailable: AMADEUS_API_KEY and AMADEUS_API_SECRET not set. Please configure them in Settings > API Keys."

    token = await _get_amadeus_token()
    if not token:
        return "Couldn't connect to hotel search service. Try again later?"

    try:
        coords = await _geocode_location(location)
        if not coords:
            return f"Couldn't find location: {location}. Please be more specific?"

        logger.info(f"Hotel search: {location} ({check_in} to {check_out})")

        headers = {"Authorization": f"Bearer {token}"}

        # Step 1: Find hotels by geocode
        url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-geocode"
        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "radius": 5,
            "radiusUnit": "KM",
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers, timeout=15.0)
            response.raise_for_status()
            hotels_data = response.json()

        hotel_ids = [h.get("hotelId") for h in hotels_data.get("data", [])[:10]]
        if not hotel_ids:
            return f"No hotels found in {location}. Try a different area?"

        # Step 2: Get offers for those hotels
        offers_url = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
        offers_params = {
            "hotelIds": ",".join(hotel_ids[:5]),
            "checkInDate": check_in,
            "checkOutDate": check_out,
            "adults": "1",
            "currency": "USD",
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(offers_url, params=offers_params, headers=headers, timeout=20.0)
            response.raise_for_status()
            offers_data = response.json()

        offers = offers_data.get("data", [])
        if not offers:
            return f"No available hotels in {location} for {check_in} to {check_out}. Try different dates?"

        lines = [f"Hotels in {location} ({check_in} to {check_out}):\n"]
        for i, offer in enumerate(offers[:5], 1):
            hotel = offer.get("hotel", {})
            hotel_offers = offer.get("offers", [])
            if not hotel_offers:
                continue
            best = hotel_offers[0]
            name = hotel.get("name", "Unknown Hotel")
            price = best.get("price", {}).get("total", "N/A")
            currency = best.get("price", {}).get("currency", "USD")
            rating = hotel.get("rating", "N/A")
            lines.append(f"{i}. {name}")
            lines.append(f"   Price: {currency} {price}/night")
            lines.append(f"   Rating: {rating}/5")
            lines.append("")

        return "\n".join(lines).strip()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            return "Invalid search: please check your dates."
        elif e.response.status_code == 401:
            return "Hotel search authentication failed. Please contact support."
        return "Couldn't search hotels. Try again later?"
    except Exception as e:
        logger.error(f"Hotel search failed: {e}", exc_info=True)
        return "Couldn't search hotels. Try again later?"


# =============================================================================
# check_weather
# =============================================================================

@tool
async def check_weather(
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
                params = {"key": api_key, "q": location, "days": days + 1, "aqi": "no"}
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
