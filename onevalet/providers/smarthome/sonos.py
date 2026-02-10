"""
Sonos Provider - Sonos Cloud Control API implementation

Uses Sonos Cloud Control API v1 for speaker and playback control.
"""

import base64
import logging
import os
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta, timezone

import httpx

from .base import BaseSmartHomeProvider

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.ws.sonos.com/control/api/v1"
TOKEN_URL = "https://api.sonos.com/login/v3/oauth/access"


class SonosProvider(BaseSmartHomeProvider):
    """Sonos speaker provider implementation using Cloud Control API v1."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self.api_base_url = API_BASE_URL

    async def list_households(self) -> Dict[str, Any]:
        """List all Sonos households associated with the account."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/households",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/households",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Sonos API error: {response.status_code}"}

                data = response.json()
                households = [
                    {"id": h["id"], "name": h.get("name", "")}
                    for h in data.get("households", [])
                ]

                logger.info(f"Sonos listed {len(households)} households")
                return {"success": True, "data": households}

        except Exception as e:
            logger.error(f"Sonos list households error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _get_first_household_id(self) -> Optional[str]:
        """Get the first household ID for convenience methods."""
        result = await self.list_households()
        if result.get("success") and result.get("data"):
            return result["data"][0]["id"]
        return None

    async def list_players(self, household_id: Optional[str] = None) -> Dict[str, Any]:
        """List all players in a household.

        Args:
            household_id: Household to query. If None, uses the first household.
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            if not household_id:
                household_id = await self._get_first_household_id()
                if not household_id:
                    return {"success": False, "error": "No households found"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/households/{household_id}/groups",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/households/{household_id}/groups",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Sonos API error: {response.status_code}"}

                data = response.json()
                players = []
                for group in data.get("groups", []):
                    coordinator_id = group.get("coordinatorId", "")
                    for player_id in group.get("playerIds", []):
                        player_info = next(
                            (p for p in data.get("players", []) if p["id"] == player_id),
                            None,
                        )
                        players.append({
                            "id": player_id,
                            "name": player_info.get("name", "") if player_info else "",
                            "group_id": group["id"],
                            "is_coordinator": player_id == coordinator_id,
                        })

                logger.info(f"Sonos listed {len(players)} players in household {household_id}")
                return {"success": True, "data": players}

        except Exception as e:
            logger.error(f"Sonos list players error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_playback_status(self, group_id: str) -> Dict[str, Any]:
        """Get the current playback status for a group.

        Args:
            group_id: The group ID to query.
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/groups/{group_id}/playback",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/groups/{group_id}/playback",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Sonos API error: {response.status_code}"}

                data = response.json()
                container = data.get("container", {})
                current_item = data.get("currentItem", {})
                track = current_item.get("track", {})

                return {
                    "success": True,
                    "data": {
                        "state": data.get("playbackState", "IDLE"),
                        "track": track.get("name", container.get("name", "")),
                        "artist": track.get("artist", {}).get("name", ""),
                        "album": track.get("album", {}).get("name", ""),
                    },
                }

        except Exception as e:
            logger.error(f"Sonos get playback status error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def play(self, group_id: str) -> Dict[str, Any]:
        """Start or resume playback for a group."""
        return await self._playback_command(group_id, "play")

    async def pause(self, group_id: str) -> Dict[str, Any]:
        """Pause playback for a group."""
        return await self._playback_command(group_id, "pause")

    async def skip_to_next(self, group_id: str) -> Dict[str, Any]:
        """Skip to the next track in a group."""
        return await self._playback_command(group_id, "skipToNextTrack")

    async def skip_to_previous(self, group_id: str) -> Dict[str, Any]:
        """Skip to the previous track in a group."""
        return await self._playback_command(group_id, "skipToPreviousTrack")

    async def _playback_command(self, group_id: str, command: str) -> Dict[str, Any]:
        """Execute a playback command on a group.

        Args:
            group_id: The group to control.
            command: One of play, pause, skipToNextTrack, skipToPreviousTrack.
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/groups/{group_id}/playback/{command}",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.post(
                            f"{self.api_base_url}/groups/{group_id}/playback/{command}",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code not in (200, 204):
                    return {"success": False, "error": f"Sonos API error: {response.status_code}"}

                logger.info(f"Sonos {command} executed on group {group_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Sonos {command} error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def set_volume(self, player_id: str, volume: int) -> Dict[str, Any]:
        """Set volume for a specific player.

        Args:
            player_id: The player to control.
            volume: Volume level 0-100.
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            volume = max(0, min(100, volume))

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/players/{player_id}/playerVolume",
                    headers=self._auth_headers(),
                    json={"volume": volume},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.post(
                            f"{self.api_base_url}/players/{player_id}/playerVolume",
                            headers=self._auth_headers(),
                            json={"volume": volume},
                            timeout=30.0,
                        )

                if response.status_code not in (200, 204):
                    return {"success": False, "error": f"Sonos API error: {response.status_code}"}

                logger.info(f"Sonos set volume to {volume} on player {player_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Sonos set volume error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_volume(self, player_id: str) -> Dict[str, Any]:
        """Get the current volume for a player.

        Args:
            player_id: The player to query.
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/players/{player_id}/playerVolume",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/players/{player_id}/playerVolume",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Sonos API error: {response.status_code}"}

                data = response.json()
                logger.info(f"Sonos got volume for player {player_id}")
                return {
                    "success": True,
                    "data": {
                        "volume": data.get("volume", 0),
                        "muted": data.get("muted", False),
                    },
                }

        except Exception as e:
            logger.error(f"Sonos get volume error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def set_mute(self, player_id: str, muted: bool) -> Dict[str, Any]:
        """Set mute state for a specific player.

        Args:
            player_id: The player to control.
            muted: True to mute, False to unmute.
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/players/{player_id}/playerVolume",
                    headers=self._auth_headers(),
                    json={"muted": muted},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.post(
                            f"{self.api_base_url}/players/{player_id}/playerVolume",
                            headers=self._auth_headers(),
                            json={"muted": muted},
                            timeout=30.0,
                        )

                if response.status_code not in (200, 204):
                    return {"success": False, "error": f"Sonos API error: {response.status_code}"}

                logger.info(f"Sonos set mute={muted} on player {player_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Sonos set mute error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def list_favorites(self, household_id: Optional[str] = None) -> Dict[str, Any]:
        """List favorites for a household.

        Args:
            household_id: Household to query. If None, uses the first household.
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            if not household_id:
                household_id = await self._get_first_household_id()
                if not household_id:
                    return {"success": False, "error": "No households found"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/households/{household_id}/favorites",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/households/{household_id}/favorites",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Sonos API error: {response.status_code}"}

                data = response.json()
                favorites = [
                    {
                        "id": f["id"],
                        "name": f.get("name", ""),
                        "type": f.get("service", {}).get("name", ""),
                    }
                    for f in data.get("items", [])
                ]

                logger.info(f"Sonos listed {len(favorites)} favorites in household {household_id}")
                return {"success": True, "data": favorites}

        except Exception as e:
            logger.error(f"Sonos list favorites error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def play_favorite(self, favorite_id: str, group_id: str) -> Dict[str, Any]:
        """Play a favorite on a group.

        Args:
            favorite_id: The favorite to play.
            group_id: The group to play on.
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/groups/{group_id}/favorites",
                    headers=self._auth_headers(),
                    json={"favoriteId": favorite_id},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.post(
                            f"{self.api_base_url}/groups/{group_id}/favorites",
                            headers=self._auth_headers(),
                            json={"favoriteId": favorite_id},
                            timeout=30.0,
                        )

                if response.status_code not in (200, 204):
                    return {"success": False, "error": f"Sonos API error: {response.status_code}"}

                logger.info(f"Sonos playing favorite {favorite_id} on group {group_id}")
                return {"success": True}

        except Exception as e:
            logger.error(f"Sonos play favorite error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_default_group(self) -> Optional[str]:
        """Get the first group ID for quick commands.

        Returns the first group's ID from the first household, or None.
        """
        try:
            household_id = await self._get_first_household_id()
            if not household_id:
                return None

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/households/{household_id}/groups",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code != 200:
                    return None

                data = response.json()
                groups = data.get("groups", [])
                if groups:
                    return groups[0]["id"]

        except Exception as e:
            logger.error(f"Sonos get default group error: {e}", exc_info=True)

        return None

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh the Sonos OAuth2 access token using the refresh token."""
        try:
            client_id = os.environ.get("SONOS_CLIENT_ID", "")
            client_secret = os.environ.get("SONOS_CLIENT_SECRET", "")

            if not client_id or not client_secret:
                return {"success": False, "error": "SONOS_CLIENT_ID and SONOS_CLIENT_SECRET env vars required"}

            credentials_str = f"{client_id}:{client_secret}"
            basic_auth = base64.b64encode(credentials_str.encode()).decode()

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
                    return {"success": False, "error": f"Token refresh failed: {response.status_code} - {response.text}"}

                data = response.json()
                expires_in = data.get("expires_in", 3600)
                token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

                if data.get("refresh_token"):
                    self.refresh_token = data["refresh_token"]
                    self.credentials["refresh_token"] = self.refresh_token

                logger.info(f"Sonos token refreshed, expires in {expires_in}s")
                return {
                    "success": True,
                    "access_token": data["access_token"],
                    "token_expiry": token_expiry,
                }

        except Exception as e:
            logger.error(f"Sonos token refresh error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
