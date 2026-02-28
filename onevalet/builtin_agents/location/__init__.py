"""
Location-based tools for OneValet.

Provides tools for querying the user's current location and setting
location-based (geofence) reminders via the koiai backend API.
"""

import logging
from typing import Optional

import httpx

from onevalet.models import AgentToolContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

KOIAI_URL: Optional[str] = None
SERVICE_KEY: Optional[str] = None


def configure(koiai_url: str = None, service_key: str = None):
    """Configure the location tools with backend URL and service key."""
    global KOIAI_URL, SERVICE_KEY
    KOIAI_URL = koiai_url
    SERVICE_KEY = service_key


# =============================================================================
# get_user_location
# =============================================================================

async def get_user_location_executor(args: dict, context: AgentToolContext = None) -> str:
    """Get the user's current location (latitude, longitude, and place name)."""
    if not context:
        return "User's location is not available. Location tracking may not be enabled."

    meta = context.metadata or {}
    location = meta.get("location")
    if location and isinstance(location, dict):
        lat = location.get("lat")
        lng = location.get("lng")
        place = location.get("place_name", "")
        if lat is not None and lng is not None:
            try:
                result = f"User is at coordinates ({float(lat):.4f}, {float(lng):.4f})"
            except (TypeError, ValueError):
                result = f"User is at coordinates ({lat}, {lng})"
            if place:
                result += f" near {place}"
            return result

    return "User's location is not available. Location tracking may not be enabled."


GET_USER_LOCATION_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}


# =============================================================================
# set_location_reminder
# =============================================================================

async def set_location_reminder_executor(args: dict, context: AgentToolContext = None) -> str:
    """Set a location-based reminder that notifies the user when they arrive near a specific place."""
    name = args.get("name", "")
    lat = args.get("lat")
    lng = args.get("lng")
    message = args.get("message", "")
    radius_meters = args.get("radius_meters", 200)

    if not name or lat is None or lng is None or not message:
        return "Error: name, lat, lng, and message are all required."

    tenant_id = (context.tenant_id if context else None) or "default"
    meta = (context.metadata if context else None) or {}

    koiai_url = KOIAI_URL or meta.get("koiai_url")
    service_key = SERVICE_KEY or meta.get("service_key", "")

    if not koiai_url:
        return "Cannot create location reminder: backend URL not configured."

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{koiai_url}/api/location/geofences",
                json={
                    "user_id": tenant_id,
                    "name": name,
                    "lat": lat,
                    "lng": lng,
                    "radius_meters": radius_meters,
                    "message": message,
                },
                headers={"X-Service-Key": service_key},
            )
            if resp.is_success:
                return (
                    f"Location reminder set! I'll notify you with '{message}' "
                    f"when you're within {radius_meters}m of {name}."
                )
            else:
                logger.error(f"Failed to create geofence: {resp.status_code} {resp.text}")
                return "Failed to create location reminder. Please try again."
    except Exception as e:
        logger.error(f"Failed to create geofence: {e}")
        return "Failed to create location reminder due to a network error."


SET_LOCATION_REMINDER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Short name for the location (e.g., 'Trader Joe\\'s', 'Office', 'Home')",
        },
        "lat": {
            "type": "number",
            "description": "Latitude of the target location",
        },
        "lng": {
            "type": "number",
            "description": "Longitude of the target location",
        },
        "message": {
            "type": "string",
            "description": "The reminder message to show the user when they arrive",
        },
        "radius_meters": {
            "type": "integer",
            "description": "Trigger radius in meters (default 200)",
            "default": 200,
        },
    },
    "required": ["name", "lat", "lng", "message"],
}
