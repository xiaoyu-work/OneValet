"""
Maps Tools — Standalone API functions for MapsAgent's mini ReAct loop.

Extracted from MapSearchAgent, DirectionsAgent, and AirQualityAgent.
Each function takes (args: dict, context: AgentToolContext) -> str.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Annotated, Any, Dict, Optional

import httpx

from koa.models import AgentToolContext, ToolOutput
from koa.tool_decorator import tool

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================


async def _geocode_location(location: str) -> Optional[Dict[str, Any]]:
    """Convert location name to coordinates using Google Geocoding API."""
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
            logger.error(
                "Geocoding returned non-OK: status=%s error_message=%s location=%r",
                data.get("status"),
                data.get("error_message"),
                location,
            )
            return None

        result = data["results"][0]
        coords = result["geometry"]["location"]
        return {
            "lat": coords["lat"],
            "lng": coords["lng"],
            "formatted_address": result["formatted_address"],
        }
    except httpx.HTTPStatusError as e:
        logger.error(
            "Geocoding HTTP error: status=%s body=%s location=%r",
            e.response.status_code,
            e.response.text[:500],
            location,
        )
        return None
    except Exception as e:
        logger.error("Geocoding failed: %s location=%r", e, location, exc_info=True)
        return None


def _parse_coords(location: str) -> Optional[tuple]:
    """Try to parse 'lat,lng' from a location string. Returns (lat, lng) or None."""
    if not location:
        return None
    m = re.match(r"^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$", location.strip())
    if m:
        lat, lng = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return (lat, lng)
    return None


# =============================================================================
# search_places
# =============================================================================


@tool
async def search_places(
    query: Annotated[str, "What to search for (e.g., 'coffee shops', 'pizza', 'gas station')"],
    location: Annotated[
        str,
        "Where to search — use 'lat,lng' coordinates (e.g., '47.7148,-122.1826') or a city/neighborhood name (e.g., 'Seattle')",
    ] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Search for places, restaurants, attractions, or businesses. Returns names, addresses, ratings, and contact info."""
    if not query:
        return "Error: query is required."

    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return "Google Maps API key not configured. Please contact support."

    try:
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": (
                "places.displayName,places.formattedAddress,places.rating,"
                "places.userRatingCount,places.types,places.priceLevel,"
                "places.businessStatus,places.googleMapsUri,"
                "places.internationalPhoneNumber,places.regularOpeningHours,"
                "places.websiteUri"
            ),
        }

        # Detect if location is lat,lng coordinates
        coords = _parse_coords(location)

        if coords:
            # Use locationBias for coordinate-based search (no "in ..." suffix)
            request_body = {
                "textQuery": query,
                "maxResultCount": 5,
                "languageCode": "en",
                "locationBias": {
                    "circle": {
                        "center": {"latitude": coords[0], "longitude": coords[1]},
                        "radius": 10000.0,  # 10 km radius
                    }
                },
            }
        else:
            text_query = f"{query} in {location}" if location else query
            request_body = {
                "textQuery": text_query,
                "maxResultCount": 5,
                "languageCode": "en",
            }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=request_body, timeout=15.0)
            response.raise_for_status()
            data = response.json()

        places = data.get("places", [])
        if not places:
            return f'No results found for "{query}" in "{location}".'

        result_lines = [f'Found {len(places)} results for "{query}" in "{location}":\n']
        place_cards = []

        for i, place in enumerate(places, 1):
            name = place.get("displayName", {}).get("text", "Unknown")
            address = place.get("formattedAddress", "")
            rating = place.get("rating")
            rating_count = place.get("userRatingCount", 0)
            phone = place.get("internationalPhoneNumber", "")
            maps_uri = place.get("googleMapsUri", "")
            website = place.get("websiteUri", "")
            price_level = place.get("priceLevel", "")
            # Map Google API price level enums to user-friendly symbols
            _PRICE_MAP = {
                "PRICE_LEVEL_FREE": "Free",
                "PRICE_LEVEL_INEXPENSIVE": "$",
                "PRICE_LEVEL_MODERATE": "$$",
                "PRICE_LEVEL_EXPENSIVE": "$$$",
                "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
            }
            price_display = _PRICE_MAP.get(price_level, price_level)
            hours = place.get("regularOpeningHours", {})
            hours_descriptions = hours.get("weekdayDescriptions", []) if hours else []
            hours_text = "; ".join(hours_descriptions) if hours_descriptions else ""

            # Extract today's hours for the card (compact display)
            today_hours = ""
            if hours_descriptions:
                weekday = datetime.now().strftime("%A")  # e.g. "Monday"
                for desc in hours_descriptions:
                    if desc.startswith(weekday):
                        # "Monday: 10:00 AM – 9:00 PM" → "10:00 AM – 9:00 PM"
                        today_hours = desc.split(": ", 1)[1] if ": " in desc else desc
                        break

            result_lines.append(f"{i}. {name}")
            if address:
                result_lines.append(f"   Address: {address}")
            if rating:
                result_lines.append(f"   Rating: {rating}/5 ({rating_count} reviews)")
            if price_display:
                result_lines.append(f"   Price: {price_display}")
            if phone:
                result_lines.append(f"   Phone: {phone}")
            if hours_text:
                result_lines.append(f"   Hours: {hours_text}")
            if maps_uri:
                result_lines.append(f"   Google Maps: {maps_uri}")
            if website:
                result_lines.append(f"   Website: {website}")
            result_lines.append("")

            # Build structured card for frontend rendering
            card: Dict[str, Any] = {
                "card_type": "place",
                "name": name,
            }
            if rating:
                card["rating"] = str(rating)
            if rating_count:
                card["reviews"] = str(rating_count)
            if address:
                card["address"] = address
            if price_display:
                card["price"] = price_display
            if phone:
                card["phone"] = phone
            if today_hours:
                card["hours"] = today_hours
            if maps_uri:
                card["mapUrl"] = maps_uri
            if website:
                card["websiteUrl"] = website
            place_cards.append(card)

        text_result = "\n".join(result_lines).strip()

        # Return ToolOutput with inline_cards media for frontend rendering
        media = []
        if place_cards:
            media.append(
                {
                    "type": "inline_cards",
                    "data": json.dumps(place_cards),
                    "media_type": "application/json",
                    "metadata": {"for_storage": False},
                }
            )

        return ToolOutput(text=text_result, media=media)

    except httpx.HTTPStatusError as e:
        # Internal diagnostic log — include response body + request for debugging (not shown to user).
        logger.error(
            "Google Places API HTTP error: status=%s body=%s request_body=%s",
            e.response.status_code,
            e.response.text[:1000],
            json.dumps(request_body)[:500],
        )
        if e.response.status_code == 400:
            return "Invalid search query. Try being more specific?"
        elif e.response.status_code in [401, 403]:
            return "Google Maps API authentication failed. Please contact support."
        return f"Couldn't search for {query}. Try again later?"
    except Exception as e:
        logger.error(f"Map search failed: {e}", exc_info=True)
        return f"Couldn't search for {query}. Try again later?"


# =============================================================================
# get_directions
# =============================================================================


@tool
async def get_directions(
    origin: Annotated[
        str, "Starting location (address or place name, or 'home' to use profile address)"
    ],
    destination: Annotated[str, "Destination (address or place name)"],
    mode: Annotated[str, "Travel mode (default: driving)"] = "driving",
    *,
    context: AgentToolContext,
) -> str:
    """Get directions between two locations. Returns distance, duration, and step-by-step navigation."""
    if not origin or not destination:
        return "Error: both origin and destination are required."

    # Resolve "home" origin from user profile
    if origin.lower() in ("home", "my home"):
        user_profile = context.user_profile or {}
        addresses = user_profile.get("addresses", [])
        if isinstance(addresses, str):
            try:
                addresses = json.loads(addresses)
            except Exception:
                addresses = []
        for addr in addresses:
            if addr.get("label") == "home":
                street = addr.get("street", "")
                city = addr.get("city", "")
                state = addr.get("state", "")
                zip_code = addr.get("zip", "")
                origin = f"{street}, {city}, {state} {zip_code}".strip(", ")
                break

    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return "Google Maps API key not configured. Please contact support."

    try:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "alternatives": "false",
            "key": api_key,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=15.0)
            response.raise_for_status()
            data = response.json()

        if data["status"] != "OK":
            logger.error(f"Directions API error: {data['status']}")
            if data["status"] == "ZERO_RESULTS":
                return f"Couldn't find a route from {origin} to {destination}."
            elif data["status"] == "NOT_FOUND":
                return "One of the locations wasn't found. Please check the addresses."
            return "Couldn't get directions. Please try again."

        routes = data.get("routes", [])
        if not routes:
            return f"No route found from {origin} to {destination}."

        route = routes[0]
        leg = route["legs"][0]

        distance = leg["distance"]["text"]
        duration = leg["duration"]["text"]
        start_address = leg["start_address"]
        end_address = leg["end_address"]
        steps = leg["steps"]

        maps_link = (
            f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}"
        )

        result_lines = [
            f"Directions from {start_address} to {end_address}:",
            f"Distance: {distance}",
            f"Duration: {duration}",
            "",
            "Steps:",
        ]
        for i, step in enumerate(steps[:5], 1):
            instruction = step.get("html_instructions", "")
            clean = re.sub(r"<[^>]+>", "", instruction).strip()
            result_lines.append(f"{i}. {clean}")
        result_lines.append(f"\nGoogle Maps: {maps_link}")

        text_result = "\n".join(result_lines)

        # Build inline card for frontend rendering
        card = {
            "card_type": "directions",
            "origin": start_address,
            "destination": end_address,
            "distance": distance,
            "duration": duration,
            "mapUrl": maps_link,
        }
        media = [
            {
                "type": "inline_cards",
                "data": json.dumps([card]),
                "media_type": "application/json",
                "metadata": {"for_storage": False},
            }
        ]

        return ToolOutput(text=text_result, media=media)

    except httpx.HTTPStatusError as e:
        logger.error(f"Directions API HTTP error: {e.response.status_code}")
        if e.response.status_code == 400:
            return "Invalid location. Please check the addresses."
        elif e.response.status_code in [401, 403]:
            return "Google Maps API authentication failed. Please contact support."
        return "Couldn't get directions. Try again later?"
    except Exception as e:
        logger.error(f"Directions API call failed: {e}", exc_info=True)
        return "Couldn't get directions. Try again later?"


# =============================================================================
# check_air_quality
# =============================================================================


@tool
async def check_air_quality(
    location: Annotated[str, "City or location name (e.g., 'Seattle', 'Beijing')"],
    *,
    context: AgentToolContext,
) -> str:
    """Check current air quality index (AQI) for a location. Returns AQI value, category, dominant pollutant, and health advice."""
    if not location:
        return "Error: location is required."

    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return "Google Maps API key not configured. Please contact support."

    try:
        coords = await _geocode_location(location)
        if not coords or "lat" not in coords:
            return f"Couldn't find {location}. Please check the location name."

        url = "https://airquality.googleapis.com/v1/currentConditions:lookup"
        headers = {"Content-Type": "application/json"}
        request_body = {
            "location": {
                "latitude": coords["lat"],
                "longitude": coords["lng"],
            },
            "extraComputations": [
                "HEALTH_RECOMMENDATIONS",
                "DOMINANT_POLLUTANT_CONCENTRATION",
                "POLLUTANT_CONCENTRATION",
                "LOCAL_AQI",
            ],
            "languageCode": "en",
        }
        params = {"key": api_key}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, headers=headers, json=request_body, params=params, timeout=15.0
            )
            response.raise_for_status()
            data = response.json()

        indexes = data.get("indexes", [])
        if not indexes:
            return f"No air quality data available for {location}."

        primary_index = indexes[0]
        aqi_value = primary_index.get("aqi")
        category = primary_index.get("category", "Unknown")
        dominant_pollutant = primary_index.get("dominantPollutant", "Unknown")

        health_recommendations = data.get("healthRecommendations", {})
        general_population = health_recommendations.get(
            "generalPopulation", "No recommendations available"
        )

        formatted_location = coords.get("formatted_address", location)

        result_lines = [
            f"Air Quality: {formatted_location}",
            f"AQI: {aqi_value}",
            f"Category: {category}",
            f"Dominant Pollutant: {dominant_pollutant}",
            f"Health Advice: {general_population}",
        ]

        text_result = "\n".join(result_lines)

        # Build inline card for frontend rendering
        card = {
            "card_type": "air_quality",
            "location": formatted_location,
            "aqi": aqi_value,
            "category": category,
            "pollutant": dominant_pollutant,
            "healthAdvice": general_population,
        }
        media = [
            {
                "type": "inline_cards",
                "data": json.dumps([card]),
                "media_type": "application/json",
                "metadata": {"for_storage": False},
            }
        ]

        return ToolOutput(text=text_result, media=media)

    except httpx.HTTPStatusError as e:
        logger.error(f"Air Quality API HTTP error: {e.response.status_code}")
        if e.response.status_code == 400:
            return f"Invalid location: {location}. Please try again."
        elif e.response.status_code in [401, 403]:
            return "Air Quality API authentication failed. Please contact support."
        return f"Couldn't get air quality for {location}. Try again later?"
    except Exception as e:
        logger.error(f"Air quality API call failed: {e}", exc_info=True)
        return "Couldn't get air quality data. Try again later?"
