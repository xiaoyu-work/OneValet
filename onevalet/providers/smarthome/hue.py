"""
Philips Hue Provider - Hue Remote API v2 implementation

Uses the Hue Remote API (CLIP v2) for controlling Philips Hue lights,
rooms, and scenes via the cloud bridge at https://api.meethue.com.
"""

import base64
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from .base import BaseSmartHomeProvider

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.meethue.com/route"
TOKEN_URL = "https://api.meethue.com/v2/oauth2/token"


class PhilipsHueProvider(BaseSmartHomeProvider):
    """Philips Hue smart home provider using Hue Remote API v2 (CLIP v2)."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self.api_base_url = API_BASE_URL

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rgb_to_xy(r: int, g: int, b: int) -> Tuple[float, float]:
        """Convert RGB (0-255) to CIE 1931 xy colour space for Hue API."""
        # Normalise to 0-1
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0

        # Apply gamma correction
        r_lin = ((r_norm + 0.055) / 1.055) ** 2.4 if r_norm > 0.04045 else r_norm / 12.92
        g_lin = ((g_norm + 0.055) / 1.055) ** 2.4 if g_norm > 0.04045 else g_norm / 12.92
        b_lin = ((b_norm + 0.055) / 1.055) ** 2.4 if b_norm > 0.04045 else b_norm / 12.92

        # Wide RGB D65 conversion
        x = r_lin * 0.664511 + g_lin * 0.154324 + b_lin * 0.162028
        y = r_lin * 0.283881 + g_lin * 0.668433 + b_lin * 0.047685
        z = r_lin * 0.000088 + g_lin * 0.072310 + b_lin * 0.986039

        total = x + y + z
        if total == 0:
            return (0.0, 0.0)

        cx = round(x / total, 4)
        cy = round(y / total, 4)
        return (cx, cy)

    async def _find_room_by_name(self, name: str) -> Optional[dict]:
        """Search rooms case-insensitively and return the first match."""
        result = await self.list_rooms()
        if not result.get("success"):
            return None
        name_lower = name.lower()
        for room in result["data"]:
            if room["name"].lower() == name_lower:
                return room
        return None

    async def _get_lights_in_room(self, room_id: str) -> List[str]:
        """Get light IDs belonging to a room."""
        result = await self.list_rooms()
        if not result.get("success"):
            return []
        for room in result["data"]:
            if room["id"] == room_id:
                return room.get("lights", [])
        return []

    # ------------------------------------------------------------------
    # Light operations
    # ------------------------------------------------------------------

    async def list_lights(self) -> Dict[str, Any]:
        """List all lights via Hue CLIP v2 API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/clip/v2/resource/light",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/clip/v2/resource/light",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                raw_lights = response.json().get("data", [])
                lights = []
                for light in raw_lights:
                    metadata = light.get("metadata", {})
                    on_state = light.get("on", {})
                    dimming = light.get("dimming", {})
                    color_temp = light.get("color_temperature", {})
                    owner = light.get("owner", {})
                    lights.append({
                        "id": light.get("id", ""),
                        "name": metadata.get("name", ""),
                        "on": on_state.get("on", False),
                        "brightness": dimming.get("brightness", 0.0),
                        "color_temp": color_temp.get("mirek", 0),
                        "room": owner.get("rid", ""),
                    })

                logger.info(f"Hue listed {len(lights)} lights")
                return {"success": True, "data": lights}

        except Exception as e:
            logger.error(f"Hue list lights error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_light_status(self, light_id: str) -> Dict[str, Any]:
        """Get detailed status for a single light."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                data = response.json().get("data", [])
                if not data:
                    return {"success": False, "error": f"Light {light_id} not found"}

                light = data[0]
                metadata = light.get("metadata", {})
                on_state = light.get("on", {})
                dimming = light.get("dimming", {})
                color_temp = light.get("color_temperature", {})
                color = light.get("color", {})
                owner = light.get("owner", {})

                logger.info(f"Hue fetched status for light {light_id}")
                return {
                    "success": True,
                    "data": {
                        "id": light.get("id", ""),
                        "name": metadata.get("name", ""),
                        "on": on_state.get("on", False),
                        "brightness": dimming.get("brightness", 0.0),
                        "color_temp": color_temp.get("mirek", 0),
                        "color_xy": color.get("xy", {}),
                        "room": owner.get("rid", ""),
                        "type": light.get("type", ""),
                    },
                }

        except Exception as e:
            logger.error(f"Hue get light status error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def turn_on(self, light_id: str) -> Dict[str, Any]:
        """Turn a light on."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                    headers=self._auth_headers(),
                    json={"on": {"on": True}},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.put(
                            f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                            headers=self._auth_headers(),
                            json={"on": {"on": True}},
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                logger.info(f"Hue turned on light {light_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Hue turn on error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def turn_off(self, light_id: str) -> Dict[str, Any]:
        """Turn a light off."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                    headers=self._auth_headers(),
                    json={"on": {"on": False}},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.put(
                            f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                            headers=self._auth_headers(),
                            json={"on": {"on": False}},
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                logger.info(f"Hue turned off light {light_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Hue turn off error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def set_brightness(self, light_id: str, brightness: float) -> Dict[str, Any]:
        """Set light brightness (0-100)."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            brightness = max(0.0, min(100.0, brightness))

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                    headers=self._auth_headers(),
                    json={"dimming": {"brightness": brightness}},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.put(
                            f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                            headers=self._auth_headers(),
                            json={"dimming": {"brightness": brightness}},
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                logger.info(f"Hue set brightness to {brightness} for light {light_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Hue set brightness error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def set_color(self, light_id: str, r: int, g: int, b: int) -> Dict[str, Any]:
        """Set light colour using RGB values (0-255). Converts to CIE xy for Hue."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            x, y = self._rgb_to_xy(r, g, b)

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                    headers=self._auth_headers(),
                    json={"color": {"xy": {"x": x, "y": y}}},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.put(
                            f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                            headers=self._auth_headers(),
                            json={"color": {"xy": {"x": x, "y": y}}},
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                logger.info(f"Hue set color to RGB({r},{g},{b}) -> xy({x},{y}) for light {light_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Hue set color error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def set_color_temperature(self, light_id: str, mirek: int) -> Dict[str, Any]:
        """Set light colour temperature in mirek (153-500, lower=cooler)."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            mirek = max(153, min(500, mirek))

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                    headers=self._auth_headers(),
                    json={"color_temperature": {"mirek": mirek}},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.put(
                            f"{self.api_base_url}/clip/v2/resource/light/{light_id}",
                            headers=self._auth_headers(),
                            json={"color_temperature": {"mirek": mirek}},
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                logger.info(f"Hue set color temperature to {mirek} mirek for light {light_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Hue set color temperature error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Room operations
    # ------------------------------------------------------------------

    async def list_rooms(self) -> Dict[str, Any]:
        """List all rooms via Hue CLIP v2 API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/clip/v2/resource/room",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/clip/v2/resource/room",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                raw_rooms = response.json().get("data", [])
                rooms = []
                for room in raw_rooms:
                    metadata = room.get("metadata", {})
                    children = room.get("children", [])
                    light_ids = [
                        child["rid"]
                        for child in children
                        if child.get("rtype") == "device"
                    ]
                    rooms.append({
                        "id": room.get("id", ""),
                        "name": metadata.get("name", ""),
                        "lights": light_ids,
                    })

                logger.info(f"Hue listed {len(rooms)} rooms")
                return {"success": True, "data": rooms}

        except Exception as e:
            logger.error(f"Hue list rooms error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def control_room(self, room_name: str, action: str, **kwargs: Any) -> Dict[str, Any]:
        """
        High-level room control: find room by name, apply action to all its lights.

        Args:
            room_name: Room name (case-insensitive match).
            action: One of "on", "off", "brightness", "color", "color_temperature".
            **kwargs: Additional args forwarded to the action method
                      (e.g. brightness=80, r=255, g=0, b=0, mirek=300).
        """
        try:
            room = await self._find_room_by_name(room_name)
            if not room:
                return {"success": False, "error": f"Room '{room_name}' not found"}

            light_ids = room.get("lights", [])
            if not light_ids:
                return {"success": False, "error": f"No lights found in room '{room_name}'"}

            results: List[Dict[str, Any]] = []
            for lid in light_ids:
                if action == "on":
                    result = await self.turn_on(lid)
                elif action == "off":
                    result = await self.turn_off(lid)
                elif action == "brightness":
                    result = await self.set_brightness(lid, kwargs.get("brightness", 100.0))
                elif action == "color":
                    result = await self.set_color(
                        lid,
                        kwargs.get("r", 255),
                        kwargs.get("g", 255),
                        kwargs.get("b", 255),
                    )
                elif action == "color_temperature":
                    result = await self.set_color_temperature(lid, kwargs.get("mirek", 300))
                else:
                    result = {"success": False, "error": f"Unknown action: {action}"}
                results.append(result)

            all_ok = all(r.get("success") for r in results)
            if all_ok:
                logger.info(f"Hue applied '{action}' to all lights in room '{room_name}'")
                return {"success": True}
            else:
                errors = [r.get("error", "") for r in results if not r.get("success")]
                return {"success": False, "error": f"Some lights failed: {'; '.join(errors)}"}

        except Exception as e:
            logger.error(f"Hue control room error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Scene operations
    # ------------------------------------------------------------------

    async def list_scenes(self) -> Dict[str, Any]:
        """List all scenes via Hue CLIP v2 API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/clip/v2/resource/scene",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/clip/v2/resource/scene",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                raw_scenes = response.json().get("data", [])
                scenes = []
                for scene in raw_scenes:
                    metadata = scene.get("metadata", {})
                    group = scene.get("group", {})
                    scenes.append({
                        "id": scene.get("id", ""),
                        "name": metadata.get("name", ""),
                        "room_id": group.get("rid", ""),
                    })

                logger.info(f"Hue listed {len(scenes)} scenes")
                return {"success": True, "data": scenes}

        except Exception as e:
            logger.error(f"Hue list scenes error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def activate_scene(self, scene_id: str) -> Dict[str, Any]:
        """Activate a scene by ID."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.api_base_url}/clip/v2/resource/scene/{scene_id}",
                    headers=self._auth_headers(),
                    json={"recall": {"action": "active"}},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("401 Unauthorized - attempting token refresh for philips_hue")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.put(
                            f"{self.api_base_url}/clip/v2/resource/scene/{scene_id}",
                            headers=self._auth_headers(),
                            json={"recall": {"action": "active"}},
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Hue API error: {response.status_code}"}

                logger.info(f"Hue activated scene {scene_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Hue activate scene error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    async def refresh_access_token(self) -> Dict[str, Any]:
        """
        Refresh Hue OAuth2 access token using refresh_token grant.

        Uses Basic auth with base64-encoded client_id:client_secret.
        Client credentials are read from HUE_CLIENT_ID and HUE_CLIENT_SECRET
        environment variables.
        """
        try:
            client_id = os.environ.get("HUE_CLIENT_ID", "")
            client_secret = os.environ.get("HUE_CLIENT_SECRET", "")

            if not client_id or not client_secret:
                return {"success": False, "error": "HUE_CLIENT_ID or HUE_CLIENT_SECRET not set"}

            if not self.refresh_token:
                return {"success": False, "error": "No refresh token available"}

            basic_auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TOKEN_URL,
                    headers={
                        "Authorization": f"Basic {basic_auth}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.refresh_token,
                    },
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(f"Hue token refresh failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Token refresh failed: {response.status_code}"}

                token_data = response.json()
                new_access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                new_refresh_token = token_data.get("refresh_token", self.refresh_token)

                self.refresh_token = new_refresh_token
                self.credentials["refresh_token"] = new_refresh_token

                token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

                logger.info("Hue access token refreshed successfully")
                return {
                    "success": True,
                    "access_token": new_access_token,
                    "expires_in": expires_in,
                    "token_expiry": token_expiry,
                }

        except Exception as e:
            logger.error(f"Hue token refresh error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
