"""
YouTubeComposioAgent - Agent for YouTube operations via Composio.

Provides search videos, get video details, and list playlists
using the Composio OAuth proxy platform.
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

# Composio action ID constants for YouTube
_ACTION_SEARCH_VIDEOS = "YOUTUBE_SEARCH_YOU_TUBE"
_ACTION_GET_VIDEO_DETAILS = "YOUTUBE_GET_VIDEO_DETAILS_BATCH"
_ACTION_LIST_PLAYLISTS = "YOUTUBE_LIST_USER_PLAYLISTS"
_APP_NAME = "youtube"


def _check_api_key() -> str | None:
    """Return error message if Composio API key is not configured, else None."""
    if not os.getenv("COMPOSIO_API_KEY"):
        return "Error: Composio API key not configured. Please add it in Settings."
    return None


# =============================================================================
# Tool executors
# =============================================================================

@tool
async def search_videos(
    query: Annotated[str, "Search keywords (e.g. 'python tutorial')"],
    limit: Annotated[int, "Max results to return"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """Search YouTube for videos matching a query."""

    if not query:
        return "Error: query is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SEARCH_VIDEOS,
            params={"q": query, "maxResults": limit},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"YouTube videos matching '{query}':\n\n{result}"
        return f"Failed to search videos: {result}"
    except Exception as e:
        logger.error(f"YouTube search_videos failed: {e}", exc_info=True)
        return f"Error searching YouTube videos: {e}"


@tool
async def get_video_details(
    video_id: Annotated[str, "YouTube video ID (e.g. 'dQw4w9WgXcQ')"],
    *,
    context: AgentToolContext,
) -> str:
    """Get detailed information about a YouTube video by its ID."""

    if not video_id:
        return "Error: video_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_VIDEO_DETAILS,
            params={"id": video_id},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Video details for '{video_id}':\n\n{result}"
        return f"Failed to get video details: {result}"
    except Exception as e:
        logger.error(f"YouTube get_video_details failed: {e}", exc_info=True)
        return f"Error getting YouTube video details: {e}"


@tool
async def list_playlists(
    limit: Annotated[int, "Maximum number of playlists to return"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """List playlists for the connected YouTube account."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_PLAYLISTS,
            params={"maxResults": limit},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"YouTube playlists:\n\n{result}"
        return f"Failed to list playlists: {result}"
    except Exception as e:
        logger.error(f"YouTube list_playlists failed: {e}", exc_info=True)
        return f"Error listing YouTube playlists: {e}"


@tool
async def connect_youtube(
    entity_id: Annotated[str, "Entity ID for multi-user setups"] = "default",
    *,
    context: AgentToolContext,
) -> str:
    """Connect your YouTube account via OAuth. Returns a URL to complete authorization."""

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
                    f"YouTube is already connected (account ID: {conn.get('id', 'unknown')}). "
                    f"You can use the other tools to interact with YouTube."
                )

        # Initiate new connection
        data = await client.initiate_connection(app_name=_APP_NAME, entity_id=entity_id)

        redirect = data.get("redirectUrl", data.get("redirect_url", ""))
        if redirect:
            return (
                f"To connect YouTube, please open this URL in your browser:\n\n"
                f"{redirect}\n\n"
                f"After completing the authorization, the connection will be active."
            )

        conn_id = data.get("id", data.get("connectedAccountId", ""))
        status = data.get("status", "")
        if status.upper() == "ACTIVE":
            return f"Successfully connected to YouTube. Connection ID: {conn_id}"
        return f"Connection initiated for YouTube. Status: {status}."
    except Exception as e:
        logger.error(f"YouTube connect failed: {e}", exc_info=True)
        return f"Error connecting to YouTube: {e}"


# =============================================================================
# Agent
# =============================================================================

@valet(capabilities=["youtube", "video", "watch"])
class YouTubeComposioAgent(StandardAgent):
    """Search YouTube videos, get video details, and list playlists.
    Use when the user mentions YouTube, videos, or wants to search/watch
    video content."""

    max_turns = 5
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a YouTube assistant with access to YouTube tools via Composio.

Available tools:
- search_videos: Search YouTube for videos matching a query.
- get_video_details: Get detailed information about a specific video by ID.
- list_playlists: List playlists for the connected YouTube account.
- connect_youtube: Connect your YouTube account (OAuth).

Instructions:
1. If the user wants to find videos, use search_videos with a keyword query.
2. If the user wants details about a specific video, use get_video_details with the video ID.
3. If the user wants to see their playlists, use list_playlists.
4. If YouTube is not yet connected, use connect_youtube first.
5. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
6. After getting tool results, provide a clear summary to the user."""

    tools = (
        search_videos,
        get_video_details,
        list_playlists,
        connect_youtube,
    )
