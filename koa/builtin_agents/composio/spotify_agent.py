"""
SpotifyComposioAgent - Agent for Spotify operations via Composio.

Provides playback control, music search, playlist management, and now-playing
info using the Composio OAuth proxy platform.
"""

import logging
import os
from typing import Annotated, Optional

from koa import valet
from koa.models import AgentToolContext
from koa.standard_agent import StandardAgent
from koa.tool_decorator import tool

from .client import ComposioClient

logger = logging.getLogger(__name__)

# Composio action ID constants for Spotify
_ACTION_START_RESUME_PLAYBACK = "SPOTIFY_START_RESUME_PLAYBACK"
_ACTION_PAUSE_PLAYBACK = "SPOTIFY_PAUSE_PLAYBACK"
_ACTION_SEARCH_FOR_ITEM = "SPOTIFY_SEARCH_FOR_ITEM"
_ACTION_GET_PLAYLISTS = "SPOTIFY_GET_CURRENT_USER_S_PLAYLISTS"
_ACTION_ADD_ITEMS_TO_PLAYLIST = "SPOTIFY_ADD_ITEMS_TO_PLAYLIST"
_ACTION_GET_CURRENTLY_PLAYING = "SPOTIFY_GET_CURRENTLY_PLAYING_TRACK"
_ACTION_SKIP_NEXT = "SPOTIFY_SKIP_TO_NEXT"
_ACTION_SKIP_PREVIOUS = "SPOTIFY_SKIP_TO_PREVIOUS"
_ACTION_GET_RECENTLY_PLAYED = "SPOTIFY_GET_RECENTLY_PLAYED_TRACKS"
_ACTION_GET_TOP_ARTISTS = "SPOTIFY_GET_USER_S_TOP_ARTISTS"
_ACTION_GET_TOP_TRACKS = "SPOTIFY_GET_USER_S_TOP_TRACKS"
_ACTION_GET_QUEUE = "SPOTIFY_GET_THE_USER_S_QUEUE"
_ACTION_TOGGLE_SHUFFLE = "SPOTIFY_TOGGLE_PLAYBACK_SHUFFLE"
_ACTION_SET_REPEAT = "SPOTIFY_SET_REPEAT_MODE"
_ACTION_SET_VOLUME = "SPOTIFY_SET_PLAYBACK_VOLUME"
_ACTION_SAVE_TRACKS = "SPOTIFY_SAVE_TRACKS_FOR_CURRENT_USER"
_ACTION_GET_RECOMMENDATIONS = "SPOTIFY_GET_RECOMMENDATIONS"
_ACTION_GET_AVAILABLE_DEVICES = "SPOTIFY_GET_AVAILABLE_DEVICES"
_ACTION_CREATE_PLAYLIST = "SPOTIFY_CREATE_PLAYLIST"
_ACTION_GET_SAVED_TRACKS = "SPOTIFY_GET_USER_S_SAVED_TRACKS"
_APP_NAME = "spotify"


def _check_api_key() -> Optional[str]:
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
    return f"Add items to Spotify playlist?\n\nPlaylist ID: {playlist_id}\nURIs: {uris}"


async def _skip_next_preview(args: dict, context) -> str:
    return "Skip to next track on Spotify?"


async def _skip_previous_preview(args: dict, context) -> str:
    return "Skip to previous track on Spotify?"


async def _toggle_shuffle_preview(args: dict, context) -> str:
    state = args.get("state", "")
    return f"Toggle Spotify shuffle to {'on' if state else 'off'}?"


async def _set_repeat_preview(args: dict, context) -> str:
    state = args.get("state", "off")
    return f"Set Spotify repeat mode to '{state}'?"


async def _set_volume_preview(args: dict, context) -> str:
    volume = args.get("volume_percent", "")
    return f"Set Spotify volume to {volume}%?"


async def _save_tracks_preview(args: dict, context) -> str:
    ids = args.get("ids", "")
    return f"Save tracks to your Spotify library?\n\nTrack IDs: {ids}"


async def _create_playlist_preview(args: dict, context) -> str:
    name = args.get("name", "")
    public = args.get("public", True)
    return f"Create a new Spotify playlist?\n\nName: {name}\nPublic: {public}"


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
            entity_id=context.tenant_id or "default",
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
            _ACTION_PAUSE_PLAYBACK, params={}, entity_id=context.tenant_id or "default"
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
            entity_id=context.tenant_id or "default",
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
            _ACTION_GET_PLAYLISTS, params={"limit": limit}, entity_id=context.tenant_id or "default"
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
    uris: Annotated[
        str, "Comma-separated Spotify URIs to add (e.g. 'spotify:track:xxx,spotify:track:yyy')"
    ],
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
            entity_id=context.tenant_id or "default",
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
            _ACTION_GET_CURRENTLY_PLAYING, params={}, entity_id=context.tenant_id or "default"
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


@tool(needs_approval=True, risk_level="write", get_preview=_skip_next_preview)
async def skip_to_next(
    *,
    context: AgentToolContext,
) -> str:
    """Skip to the next track in the Spotify playback queue."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SKIP_NEXT,
            params={},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Skipped to next track.\n\n{result}"
        return f"Failed to skip to next track: {result}"
    except Exception as e:
        logger.error(f"Spotify skip_to_next failed: {e}", exc_info=True)
        return f"Error skipping to next track: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_skip_previous_preview)
async def skip_to_previous(
    *,
    context: AgentToolContext,
) -> str:
    """Skip to the previous track in the Spotify playback queue."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SKIP_PREVIOUS,
            params={},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Skipped to previous track.\n\n{result}"
        return f"Failed to skip to previous track: {result}"
    except Exception as e:
        logger.error(f"Spotify skip_to_previous failed: {e}", exc_info=True)
        return f"Error skipping to previous track: {e}"


@tool
async def get_recently_played(
    limit: Annotated[int, "Maximum number of recently played tracks to return"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Get the user's recently played tracks on Spotify."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_RECENTLY_PLAYED,
            params={"limit": limit},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Recently played tracks:\n\n{result}"
        return f"Failed to get recently played tracks: {result}"
    except Exception as e:
        logger.error(f"Spotify get_recently_played failed: {e}", exc_info=True)
        return f"Error getting recently played tracks: {e}"


@tool
async def get_top_artists(
    limit: Annotated[int, "Maximum number of top artists to return"] = 10,
    time_range: Annotated[
        str, "Time range: short_term (4 weeks), medium_term (6 months), or long_term (years)"
    ] = "medium_term",
    *,
    context: AgentToolContext,
) -> str:
    """Get the user's top artists on Spotify."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_TOP_ARTISTS,
            params={"limit": limit, "time_range": time_range},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Your top artists ({time_range}):\n\n{result}"
        return f"Failed to get top artists: {result}"
    except Exception as e:
        logger.error(f"Spotify get_top_artists failed: {e}", exc_info=True)
        return f"Error getting top artists: {e}"


@tool
async def get_top_tracks(
    limit: Annotated[int, "Maximum number of top tracks to return"] = 10,
    time_range: Annotated[
        str, "Time range: short_term (4 weeks), medium_term (6 months), or long_term (years)"
    ] = "medium_term",
    *,
    context: AgentToolContext,
) -> str:
    """Get the user's top tracks on Spotify."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_TOP_TRACKS,
            params={"limit": limit, "time_range": time_range},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Your top tracks ({time_range}):\n\n{result}"
        return f"Failed to get top tracks: {result}"
    except Exception as e:
        logger.error(f"Spotify get_top_tracks failed: {e}", exc_info=True)
        return f"Error getting top tracks: {e}"


@tool
async def get_queue(
    *,
    context: AgentToolContext,
) -> str:
    """Get the user's current Spotify playback queue."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_QUEUE,
            params={},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Current playback queue:\n\n{result}"
        return f"Failed to get playback queue: {result}"
    except Exception as e:
        logger.error(f"Spotify get_queue failed: {e}", exc_info=True)
        return f"Error getting playback queue: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_toggle_shuffle_preview)
async def toggle_shuffle(
    state: Annotated[bool, "True to enable shuffle, False to disable"],
    *,
    context: AgentToolContext,
) -> str:
    """Toggle shuffle mode for Spotify playback."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_TOGGLE_SHUFFLE,
            params={"state": state},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            label = "enabled" if state else "disabled"
            return f"Shuffle {label}.\n\n{result}"
        return f"Failed to toggle shuffle: {result}"
    except Exception as e:
        logger.error(f"Spotify toggle_shuffle failed: {e}", exc_info=True)
        return f"Error toggling shuffle: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_set_repeat_preview)
async def set_repeat(
    state: Annotated[
        str, "Repeat mode: 'track' (repeat current), 'context' (repeat playlist/album), or 'off'"
    ],
    *,
    context: AgentToolContext,
) -> str:
    """Set the repeat mode for Spotify playback."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SET_REPEAT,
            params={"state": state},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Repeat mode set to '{state}'.\n\n{result}"
        return f"Failed to set repeat mode: {result}"
    except Exception as e:
        logger.error(f"Spotify set_repeat failed: {e}", exc_info=True)
        return f"Error setting repeat mode: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_set_volume_preview)
async def set_volume(
    volume_percent: Annotated[int, "Volume level from 0 to 100"],
    *,
    context: AgentToolContext,
) -> str:
    """Set the playback volume for Spotify (0-100)."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SET_VOLUME,
            params={"volume_percent": volume_percent},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Volume set to {volume_percent}%.\n\n{result}"
        return f"Failed to set volume: {result}"
    except Exception as e:
        logger.error(f"Spotify set_volume failed: {e}", exc_info=True)
        return f"Error setting volume: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_save_tracks_preview)
async def save_tracks(
    ids: Annotated[str, "Comma-separated Spotify track IDs to save to the user's library"],
    *,
    context: AgentToolContext,
) -> str:
    """Save one or more tracks to the user's Spotify library (liked songs)."""

    if not ids:
        return "Error: ids is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SAVE_TRACKS,
            params={"ids": ids},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Tracks saved to your library.\n\n{result}"
        return f"Failed to save tracks: {result}"
    except Exception as e:
        logger.error(f"Spotify save_tracks failed: {e}", exc_info=True)
        return f"Error saving tracks: {e}"


@tool
async def get_recommendations(
    seed_artists: Annotated[
        str, "Comma-separated Spotify artist IDs for seeding recommendations"
    ] = "",
    seed_tracks: Annotated[
        str, "Comma-separated Spotify track IDs for seeding recommendations"
    ] = "",
    seed_genres: Annotated[
        str, "Comma-separated genre names for seeding recommendations (e.g. 'pop,rock')"
    ] = "",
    limit: Annotated[int, "Maximum number of recommendations to return"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """Get track recommendations from Spotify based on seed artists, tracks, or genres."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        params: dict = {"limit": limit}
        if seed_artists:
            params["seed_artists"] = seed_artists
        if seed_tracks:
            params["seed_tracks"] = seed_tracks
        if seed_genres:
            params["seed_genres"] = seed_genres
        data = await client.execute_action(
            _ACTION_GET_RECOMMENDATIONS,
            params=params,
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Recommended tracks:\n\n{result}"
        return f"Failed to get recommendations: {result}"
    except Exception as e:
        logger.error(f"Spotify get_recommendations failed: {e}", exc_info=True)
        return f"Error getting recommendations: {e}"


@tool
async def get_available_devices(
    *,
    context: AgentToolContext,
) -> str:
    """List the user's available Spotify playback devices."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_AVAILABLE_DEVICES,
            params={},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Available Spotify devices:\n\n{result}"
        return f"Failed to get available devices: {result}"
    except Exception as e:
        logger.error(f"Spotify get_available_devices failed: {e}", exc_info=True)
        return f"Error getting available devices: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_create_playlist_preview)
async def create_playlist(
    user_id: Annotated[str, "Spotify user ID to create the playlist for"],
    name: Annotated[str, "Name for the new playlist"],
    description: Annotated[str, "Optional description for the playlist"] = "",
    public: Annotated[bool, "Whether the playlist should be public"] = True,
    *,
    context: AgentToolContext,
) -> str:
    """Create a new Spotify playlist for the specified user."""

    if not user_id:
        return "Error: user_id is required."
    if not name:
        return "Error: name is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        params: dict = {"user_id": user_id, "name": name, "public": public}
        if description:
            params["description"] = description
        data = await client.execute_action(
            _ACTION_CREATE_PLAYLIST,
            params=params,
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Playlist '{name}' created.\n\n{result}"
        return f"Failed to create playlist: {result}"
    except Exception as e:
        logger.error(f"Spotify create_playlist failed: {e}", exc_info=True)
        return f"Error creating playlist: {e}"


@tool
async def get_saved_tracks(
    limit: Annotated[int, "Maximum number of saved tracks to return"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Get the user's saved (liked) tracks on Spotify."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_SAVED_TRACKS,
            params={"limit": limit},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Your saved tracks:\n\n{result}"
        return f"Failed to get saved tracks: {result}"
    except Exception as e:
        logger.error(f"Spotify get_saved_tracks failed: {e}", exc_info=True)
        return f"Error getting saved tracks: {e}"


# =============================================================================
# Agent
# =============================================================================


@valet(domain="lifestyle")
class SpotifyComposioAgent(StandardAgent):
    """Control Spotify playback, search music, manage playlists, and check
    what's currently playing. Use when the user mentions Spotify, music,
    songs, playlists, or playback control."""

    max_turns = 15
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
- skip_to_next: Skip to the next track in the playback queue.
- skip_to_previous: Skip to the previous track in the playback queue.
- get_recently_played: Get the user's recently played tracks.
- get_top_artists: Get the user's top artists over a time range.
- get_top_tracks: Get the user's top tracks over a time range.
- get_queue: Get the current playback queue.
- toggle_shuffle: Enable or disable shuffle mode.
- set_repeat: Set repeat mode (track, context, or off).
- set_volume: Set playback volume (0-100).
- save_tracks: Save tracks to the user's library (liked songs).
- get_recommendations: Get track recommendations based on seed artists, tracks, or genres.
- get_available_devices: List available Spotify playback devices.
- create_playlist: Create a new Spotify playlist.
- get_saved_tracks: Get the user's saved (liked) tracks.
- connect_spotify: Connect your Spotify account (OAuth).

Instructions:
1. If the user wants to play music, use play_music. If they specify a song/album/playlist, search first to get the URI, then play it.
2. If the user wants to pause, use pause_music.
3. If the user wants to find music, use search_music with a query and optional type filter.
4. If the user wants to see their playlists, use get_playlists.
5. If the user wants to add songs to a playlist, use add_to_playlist with the playlist ID and track URIs.
6. If the user wants to know what's playing, use now_playing.
7. If the user wants to skip forward or backward, use skip_to_next or skip_to_previous.
8. If the user wants playback history, use get_recently_played.
9. If the user wants their top artists or tracks, use get_top_artists or get_top_tracks.
10. If the user wants to see the queue, use get_queue.
11. If the user wants to toggle shuffle or repeat, use toggle_shuffle or set_repeat.
12. If the user wants to change volume, use set_volume.
13. If the user wants to like/save a track, use save_tracks.
14. If the user wants music recommendations, use get_recommendations with relevant seeds.
15. If the user wants to see available devices, use get_available_devices.
16. If the user wants to create a playlist, use create_playlist.
17. If the user wants to see their liked songs, use get_saved_tracks.
18. If Spotify is not yet connected, use connect_spotify first.
19. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
20. After getting tool results, provide a clear summary to the user."""

    tools = (
        play_music,
        pause_music,
        search_music,
        get_playlists,
        add_to_playlist,
        now_playing,
        skip_to_next,
        skip_to_previous,
        get_recently_played,
        get_top_artists,
        get_top_tracks,
        get_queue,
        toggle_shuffle,
        set_repeat,
        set_volume,
        save_tracks,
        get_recommendations,
        get_available_devices,
        create_playlist,
        get_saved_tracks,
        connect_spotify,
    )
