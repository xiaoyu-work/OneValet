"""
YouTubeComposioAgent - Agent for YouTube operations via Composio.

Provides search videos, get video details, list playlists, manage
subscriptions, browse channels, and retrieve captions using the
Composio OAuth proxy platform.
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

# Composio action ID constants for YouTube
_ACTION_SEARCH_VIDEOS = "YOUTUBE_SEARCH_YOU_TUBE"
_ACTION_VIDEO_DETAILS = "YOUTUBE_VIDEO_DETAILS"
_ACTION_LIST_PLAYLISTS = "YOUTUBE_LIST_USER_PLAYLISTS"
_ACTION_LIST_SUBSCRIPTIONS = "YOUTUBE_LIST_USER_SUBSCRIPTIONS"
_ACTION_LIST_CHANNEL_VIDEOS = "YOUTUBE_LIST_CHANNEL_VIDEOS"
_ACTION_GET_CHANNEL_STATS = "YOUTUBE_GET_CHANNEL_STATISTICS"
_ACTION_GET_CHANNEL_ACTIVITIES = "YOUTUBE_GET_CHANNEL_ACTIVITIES"
_ACTION_GET_CHANNEL_BY_HANDLE = "YOUTUBE_GET_CHANNEL_ID_BY_HANDLE"
_ACTION_SUBSCRIBE = "YOUTUBE_SUBSCRIBE_CHANNEL"
_ACTION_LIST_CAPTIONS = "YOUTUBE_LIST_CAPTION_TRACK"
_ACTION_LOAD_CAPTIONS = "YOUTUBE_LOAD_CAPTIONS"
_APP_NAME = "youtube"


def _check_api_key() -> Optional[str]:
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
            entity_id=context.tenant_id or "default",
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
            _ACTION_VIDEO_DETAILS,
            params={"id": video_id},
            entity_id=context.tenant_id or "default",
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
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"YouTube playlists:\n\n{result}"
        return f"Failed to list playlists: {result}"
    except Exception as e:
        logger.error(f"YouTube list_playlists failed: {e}", exc_info=True)
        return f"Error listing YouTube playlists: {e}"


@tool
async def list_subscriptions(
    limit: Annotated[int, "Maximum number of subscriptions to return"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """List channels the connected YouTube account is subscribed to."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_SUBSCRIPTIONS,
            params={"maxResults": limit},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"YouTube subscriptions:\n\n{result}"
        return f"Failed to list subscriptions: {result}"
    except Exception as e:
        logger.error(f"YouTube list_subscriptions failed: {e}", exc_info=True)
        return f"Error listing YouTube subscriptions: {e}"


@tool
async def list_channel_videos(
    channel_id: Annotated[str, "YouTube channel ID (e.g. 'UC...')"],
    limit: Annotated[int, "Maximum number of videos to return"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """List recent videos from a specific YouTube channel."""

    if not channel_id:
        return "Error: channel_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_CHANNEL_VIDEOS,
            params={"channelId": channel_id, "maxResults": limit},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Videos from channel '{channel_id}':\n\n{result}"
        return f"Failed to list channel videos: {result}"
    except Exception as e:
        logger.error(f"YouTube list_channel_videos failed: {e}", exc_info=True)
        return f"Error listing YouTube channel videos: {e}"


@tool
async def get_channel_stats(
    channel_id: Annotated[str, "YouTube channel ID (e.g. 'UC...')"],
    *,
    context: AgentToolContext,
) -> str:
    """Get statistics (subscriber count, view count, video count) for a YouTube channel."""

    if not channel_id:
        return "Error: channel_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_CHANNEL_STATS,
            params={"channelId": channel_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Channel statistics for '{channel_id}':\n\n{result}"
        return f"Failed to get channel statistics: {result}"
    except Exception as e:
        logger.error(f"YouTube get_channel_stats failed: {e}", exc_info=True)
        return f"Error getting YouTube channel statistics: {e}"


@tool
async def get_channel_activities(
    channel_id: Annotated[str, "YouTube channel ID (e.g. 'UC...')"],
    *,
    context: AgentToolContext,
) -> str:
    """Get recent activities (uploads, playlist additions) for a YouTube channel."""

    if not channel_id:
        return "Error: channel_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_CHANNEL_ACTIVITIES,
            params={"channelId": channel_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Recent activities for channel '{channel_id}':\n\n{result}"
        return f"Failed to get channel activities: {result}"
    except Exception as e:
        logger.error(f"YouTube get_channel_activities failed: {e}", exc_info=True)
        return f"Error getting YouTube channel activities: {e}"


@tool
async def get_channel_by_handle(
    handle: Annotated[str, "YouTube handle (e.g. '@mkbhd')"],
    *,
    context: AgentToolContext,
) -> str:
    """Resolve a YouTube @handle to a channel ID."""

    if not handle:
        return "Error: handle is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_CHANNEL_BY_HANDLE,
            params={"handle": handle},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Channel info for handle '{handle}':\n\n{result}"
        return f"Failed to resolve handle: {result}"
    except Exception as e:
        logger.error(f"YouTube get_channel_by_handle failed: {e}", exc_info=True)
        return f"Error resolving YouTube handle: {e}"


@tool
async def subscribe_channel(
    channel_id: Annotated[str, "YouTube channel ID to subscribe to"],
    *,
    context: AgentToolContext,
) -> str:
    """Subscribe the connected YouTube account to a channel."""

    if not channel_id:
        return "Error: channel_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SUBSCRIBE,
            params={"channelId": channel_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Subscribed to channel '{channel_id}'.\n\n{result}"
        return f"Failed to subscribe to channel: {result}"
    except Exception as e:
        logger.error(f"YouTube subscribe_channel failed: {e}", exc_info=True)
        return f"Error subscribing to YouTube channel: {e}"


@tool
async def list_captions(
    video_id: Annotated[str, "YouTube video ID (e.g. 'dQw4w9WgXcQ')"],
    *,
    context: AgentToolContext,
) -> str:
    """List available caption tracks for a YouTube video."""

    if not video_id:
        return "Error: video_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_CAPTIONS,
            params={"videoId": video_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Caption tracks for video '{video_id}':\n\n{result}"
        return f"Failed to list captions: {result}"
    except Exception as e:
        logger.error(f"YouTube list_captions failed: {e}", exc_info=True)
        return f"Error listing YouTube captions: {e}"


@tool
async def download_captions(
    caption_id: Annotated[str, "Caption track ID from list_captions"],
    *,
    context: AgentToolContext,
) -> str:
    """Download the text of a caption track. Useful for summarizing video content."""

    if not caption_id:
        return "Error: caption_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LOAD_CAPTIONS,
            params={"captionId": caption_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Caption text:\n\n{result}"
        return f"Failed to download captions: {result}"
    except Exception as e:
        logger.error(f"YouTube download_captions failed: {e}", exc_info=True)
        return f"Error downloading YouTube captions: {e}"


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


@valet(domain="lifestyle")
class YouTubeComposioAgent(StandardAgent):
    """Search YouTube videos, browse channels, manage subscriptions, get
    captions, and list playlists. Use when the user mentions YouTube,
    videos, channels, or wants to search/watch video content."""

    max_turns = 15
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a YouTube assistant with access to YouTube tools via Composio.

Available tools:
- search_videos: Search YouTube for videos matching a query.
- get_video_details: Get detailed information about a specific video by ID.
- list_playlists: List playlists for the connected YouTube account.
- list_subscriptions: List channels the user is subscribed to.
- list_channel_videos: List recent videos from a specific channel.
- get_channel_stats: Get subscriber count, view count, and video count for a channel.
- get_channel_activities: Get recent uploads and playlist additions for a channel.
- get_channel_by_handle: Resolve a YouTube @handle (e.g. @mkbhd) to a channel ID.
- subscribe_channel: Subscribe to a YouTube channel.
- list_captions: List available caption tracks for a video.
- download_captions: Download caption text for a video (useful for summarization).
- connect_youtube: Connect your YouTube account (OAuth).

Instructions:
1. If the user wants to find videos, use search_videos with a keyword query.
2. If the user wants details about a specific video, use get_video_details with the video ID.
3. If the user wants to see their playlists, use list_playlists.
4. If the user wants to see their subscriptions, use list_subscriptions.
5. If the user mentions a channel by @handle, use get_channel_by_handle first to get the channel ID.
6. If the user wants to browse a channel's videos, use list_channel_videos with the channel ID.
7. If the user wants channel stats, use get_channel_stats with the channel ID.
8. If the user wants to subscribe to a channel, use subscribe_channel.
9. If the user wants video captions or a transcript, use list_captions then download_captions.
10. If YouTube is not yet connected, use connect_youtube first.
11. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
12. After getting tool results, provide a clear summary to the user.

Response format:
- Always include clickable YouTube links for videos: https://www.youtube.com/watch?v={videoId}
- Always include channel links when relevant: https://www.youtube.com/channel/{channelId}
- Always include playlist links when relevant: https://www.youtube.com/playlist?list={playlistId}
- Format each result with title, channel name, and link on the same line."""

    tools = (
        search_videos,
        get_video_details,
        list_playlists,
        list_subscriptions,
        list_channel_videos,
        get_channel_stats,
        get_channel_activities,
        get_channel_by_handle,
        subscribe_channel,
        list_captions,
        download_captions,
        connect_youtube,
    )
