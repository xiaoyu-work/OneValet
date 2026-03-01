"""
SpotifyComposioAgent - Agent for Spotify operations via Composio.

Provides playback control, music search, playlist management, and now-playing
info using the Composio OAuth proxy platform.
"""

import os
import logging
from typing import Annotated

from onevalet import valet
from onevalet.models import AgentToolContext
from onevalet.standard_agent import StandardAgent
from onevalet.tool_decorator import tool

from .client import ComposioClient

logger = logging.getLogger(__name__)

# Composio action ID constants for Spotify
_ACTION_START_RESUME_PLAYBACK = "SPOTIFY_START_RESUME_PLAYBACK"
_ACTION_PAUSE_PLAYBACK = "SPOTIFY_PAUSE_PLAYBACK"
_ACTION_SEARCH_FOR_ITEM = "SPOTIFY_SEARCH_FOR_ITEM"
_ACTION_GET_PLAYLISTS = "SPOTIFY_GET_CURRENT_USER_S_PLAYLISTS"
_ACTION_ADD_ITEMS_TO_PLAYLIST = "SPOTIFY_ADD_ITEMS_TO_PLAYLIST"
_ACTION_GET_CURRENTLY_PLAYING = "SPOTIFY_GET_CURRENTLY_PLAYING_TRACK"
_APP_NAME = "spotify"


def _check_api_key() -> str | None:
    """Return error message if Composio API key is not configured, else None."""
    if not os.getenv("COMPOSIO_API_KEY"):
        return "Error: Composio API key not configured. Please add it in Settings."
    return None


# =============================================================================
# Approval preview functions
# =============================================================================

async def _play_music_preview(args: dict, context) -> str:
    uri = args.get("uri", "")
    device_id = args.get("device_id", "")
    parts = ["Start/resume Spotify playback?"]
    if uri:
        parts.append(f"\nURI: {uri}")
    if device_id:
        parts.append(f"\nDevice: {device_id}")
    return "".join(parts)


async def _pause_music_preview(args: dict, context) -> str:
    return "Pause Spotify playback?"


async def _add_to_playlist_preview(args: dict, context) -> str:
    playlist_id = args.get("playlist_id", "")
    uris = args.get("uris", "")
    return (
        f"Add items to Spotify playlist?\n\n"
        f"Playlist ID: {playlist_id}\n"
        f"URIs: {uris}"
    )


# =============================================================================
# Tool executors
# =============================================================================

@tool(needs_approval=True, risk_level="write", get_preview=_play_music_preview)
async def play_music(
    uri: Annotated[str, "Optional Spotify URI (track, album, or playlist) to play"] = "",
    device_id: Annotated[str, "Optional target device ID for playback"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Start or resume Spotify playback. Optionally specify a track, album, or playlist URI."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        params = {}
        if uri:
            params["uri"] = uri
        if device_id:
            params["device_id"] = device_id

        data = await client.execute_action(
            _ACTION_START_RESUME_PLAYBACK,
            params=params,
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            if uri:
                return f"Playback started for {uri}.\n\n{result}"
            return f"Playback resumed.\n\n{result}"
        return f"Failed to start/resume playback: {result}"
    except Exception as e:
        logger.error(f"Spotify play_music failed: {e}", exc_info=True)
        return f"Error starting Spotify playback: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_pause_music_preview)
async def pause_music(
    *,
    context: AgentToolContext,
) -> str:
    """Pause Spotify playback on the active device."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_PAUSE_PLAYBACK,
            params={},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Playback paused.\n\n{result}"
        return f"Failed to pause playback: {result}"
    except Exception as e:
        logger.error(f"Spotify pause_music failed: {e}", exc_info=True)
        return f"Error pausing Spotify playback: {e}"


@tool
async def search_music(
    query: Annotated[str, "Search keywords (e.g. 'Bohemian Rhapsody', 'Taylor Swift')"],
    type: Annotated[str, "Type of item to search for: track, album, artist, or playlist"] = "track",
    limit: Annotated[int, "Maximum number of results to return"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """Search Spotify for tracks, albums, artists, or playlists."""

    if not query:
        return "Error: query is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SEARCH_FOR_ITEM,
            params={"q": query, "type": type, "limit": limit},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Spotify search results for '{query}' ({type}):\n\n{result}"
        return f"Failed to search Spotify: {result}"
    except Exception as e:
        logger.error(f"Spotify search_music failed: {e}", exc_info=True)
        return f"Error searching Spotify: {e}"


@tool
async def get_playlists(
    limit: Annotated[int, "Maximum number of playlists to return"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Get the current user's Spotify playlists."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_PLAYLISTS,
            params={"limit": limit},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Your Spotify playlists:\n\n{result}"
        return f"Failed to get playlists: {result}"
    except Exception as e:
        logger.error(f"Spotify get_playlists failed: {e}", exc_info=True)
        return f"Error getting Spotify playlists: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_add_to_playlist_preview)
async def add_to_playlist(
    playlist_id: Annotated[str, "Spotify playlist ID to add items to"],
    uris: Annotated[str, "Comma-separated Spotify URIs to add (e.g. 'spotify:track:xxx,spotify:track:yyy')"],
    *,
    context: AgentToolContext,
) -> str:
    """Add one or more items to a Spotify playlist."""

    if not playlist_id:
        return "Error: playlist_id is required."
    if not uris:
        return "Error: uris is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_ADD_ITEMS_TO_PLAYLIST,
            params={"playlist_id": playlist_id, "uris": uris},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Items added to playlist {playlist_id}.\n\n{result}"
        return f"Failed to add items to playlist: {result}"
    except Exception as e:
        logger.error(f"Spotify add_to_playlist failed: {e}", exc_info=True)
        return f"Error adding items to Spotify playlist: {e}"


@tool
async def now_playing(
    *,
    context: AgentToolContext,
) -> str:
    """Get the currently playing track on Spotify."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_CURRENTLY_PLAYING,
            params={},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Currently playing on Spotify:\n\n{result}"
        return f"Failed to get currently playing track: {result}"
    except Exception as e:
        logger.error(f"Spotify now_playing failed: {e}", exc_info=True)
        return f"Error getting currently playing track: {e}"


@tool
async def connect_spotify(
    entity_id: Annotated[str, "Entity ID for multi-user setups"] = "default",
    *,
    context: AgentToolContext,
) -> str:
    """Connect your Spotify account via OAuth. Returns a URL to complete authorization."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()

        # Check for existing active connection
        connections = await client.list_connections(entity_id=entity_id)
        connection_list = connections.get("items", connections.get("connections", []))
        for conn in connection_list:
            conn_app = (conn.get("appName") or conn.get("appUniqueId") or "").lower()
            conn_status = (conn.get("status") or "").upper()
            if conn_app == _APP_NAME and conn_status == "ACTIVE":
                return (
                    f"Spotify is already connected (account ID: {conn.get('id', 'unknown')}). "
                    f"You can use the other tools to interact with Spotify."
                )

        # Initiate new connection
        data = await client.initiate_connection(app_name=_APP_NAME, entity_id=entity_id)

        redirect = data.get("redirectUrl", data.get("redirect_url", ""))
        if redirect:
            return (
                f"To connect Spotify, please open this URL in your browser:\n\n"
                f"{redirect}\n\n"
                f"After completing the authorization, the connection will be active."
            )

        conn_id = data.get("id", data.get("connectedAccountId", ""))
        status = data.get("status", "")
        if status.upper() == "ACTIVE":
            return f"Successfully connected to Spotify. Connection ID: {conn_id}"
        return f"Connection initiated for Spotify. Status: {status}."
    except Exception as e:
        logger.error(f"Spotify connect failed: {e}", exc_info=True)
        return f"Error connecting to Spotify: {e}"


# =============================================================================
# Agent
# =============================================================================

@valet(domain="lifestyle")
class SpotifyComposioAgent(StandardAgent):
    """Control Spotify playback, search music, manage playlists, and check
    what's currently playing. Use when the user mentions Spotify, music,
    songs, playlists, or playback control."""

    max_turns = 5
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a Spotify assistant with access to Spotify tools via Composio.

Available tools:
- play_music: Start or resume playback, optionally with a specific track/album/playlist URI.
- pause_music: Pause the current playback.
- search_music: Search Spotify for tracks, albums, artists, or playlists.
- get_playlists: List the current user's Spotify playlists.
- add_to_playlist: Add items (tracks) to a Spotify playlist.
- now_playing: Get the currently playing track.
- connect_spotify: Connect your Spotify account (OAuth).

Instructions:
1. If the user wants to play music, use play_music. If they specify a song/album/playlist, search first to get the URI, then play it.
2. If the user wants to pause, use pause_music.
3. If the user wants to find music, use search_music with a query and optional type filter.
4. If the user wants to see their playlists, use get_playlists.
5. If the user wants to add songs to a playlist, use add_to_playlist with the playlist ID and track URIs.
6. If the user wants to know what's playing, use now_playing.
7. If Spotify is not yet connected, use connect_spotify first.
8. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
9. After getting tool results, provide a clear summary to the user."""

    tools = (
        play_music,
        pause_music,
        search_music,
        get_playlists,
        add_to_playlist,
        now_playing,
        connect_spotify,
    )
