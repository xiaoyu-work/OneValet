"""
DiscordComposioAgent - Agent for Discord operations via Composio.

Provides send messages, list channels, and list servers/guilds
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

# Composio action ID constants for Discord
_ACTION_CREATE_MESSAGE = "DISCORDBOT_CREATE_MESSAGE"
_ACTION_LIST_GUILD_CHANNELS = "DISCORDBOT_LIST_GUILD_CHANNELS"
_ACTION_LIST_MY_GUILDS = "DISCORD_LIST_MY_GUILDS"
_APP_NAME = "discord"


def _check_api_key() -> str | None:
    """Return error message if Composio API key is not configured, else None."""
    if not os.getenv("COMPOSIO_API_KEY"):
        return "Error: Composio API key not configured. Please add it in Settings."
    return None


# =============================================================================
# Approval preview functions
# =============================================================================

async def _send_message_preview(args: dict, context) -> str:
    channel_id = args.get("channel_id", "")
    content = args.get("content", "")
    preview = content[:100] + "..." if len(content) > 100 else content
    return f"Send Discord message?\n\nChannel ID: {channel_id}\nMessage: {preview}"


# =============================================================================
# Tool executors
# =============================================================================

@tool(needs_approval=True, risk_level="write", get_preview=_send_message_preview)
async def send_message(
    channel_id: Annotated[str, "Discord channel ID to send the message to"],
    content: Annotated[str, "Message content to send"],
    *,
    context: AgentToolContext,
) -> str:
    """Send a message to a Discord channel."""

    if not channel_id:
        return "Error: channel_id is required."
    if not content:
        return "Error: content is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_CREATE_MESSAGE,
            params={"channel_id": channel_id, "content": content},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Message sent to channel {channel_id}.\n\n{result}"
        return f"Failed to send message: {result}"
    except Exception as e:
        logger.error(f"Discord send_message failed: {e}", exc_info=True)
        return f"Error sending Discord message: {e}"


@tool
async def list_channels(
    guild_id: Annotated[str, "Discord guild/server ID to list channels for"],
    *,
    context: AgentToolContext,
) -> str:
    """List all channels in a Discord guild/server."""

    if not guild_id:
        return "Error: guild_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_GUILD_CHANNELS,
            params={"guild_id": guild_id},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Channels in guild {guild_id}:\n\n{result}"
        return f"Failed to list channels: {result}"
    except Exception as e:
        logger.error(f"Discord list_channels failed: {e}", exc_info=True)
        return f"Error listing Discord channels: {e}"


@tool
async def list_servers(
    *,
    context: AgentToolContext,
) -> str:
    """List all Discord guilds/servers the user belongs to."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_MY_GUILDS,
            params={},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Discord servers:\n\n{result}"
        return f"Failed to list servers: {result}"
    except Exception as e:
        logger.error(f"Discord list_servers failed: {e}", exc_info=True)
        return f"Error listing Discord servers: {e}"


@tool
async def connect_discord(
    entity_id: Annotated[str, "Entity ID for multi-user setups"] = "default",
    *,
    context: AgentToolContext,
) -> str:
    """Connect your Discord account via OAuth. Returns a URL to complete authorization."""

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
                    f"Discord is already connected (account ID: {conn.get('id', 'unknown')}). "
                    f"You can use the other tools to interact with Discord."
                )

        # Initiate new connection
        data = await client.initiate_connection(app_name=_APP_NAME, entity_id=entity_id)

        redirect = data.get("redirectUrl", data.get("redirect_url", ""))
        if redirect:
            return (
                f"To connect Discord, please open this URL in your browser:\n\n"
                f"{redirect}\n\n"
                f"After completing the authorization, the connection will be active."
            )

        conn_id = data.get("id", data.get("connectedAccountId", ""))
        status = data.get("status", "")
        if status.upper() == "ACTIVE":
            return f"Successfully connected to Discord. Connection ID: {conn_id}"
        return f"Connection initiated for Discord. Status: {status}."
    except Exception as e:
        logger.error(f"Discord connect failed: {e}", exc_info=True)
        return f"Error connecting to Discord: {e}"


# =============================================================================
# Agent
# =============================================================================

@valet(domain="communication")
class DiscordComposioAgent(StandardAgent):
    """Send messages, list channels, and manage Discord servers. Use when
    the user mentions Discord, guilds, servers, or wants to send/read
    messages on Discord."""

    max_turns = 5
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a Discord assistant with access to Discord tools via Composio.

Available tools:
- send_message: Send a message to a Discord channel.
- list_channels: List all channels in a Discord guild/server.
- list_servers: List all Discord guilds/servers you belong to.
- connect_discord: Connect your Discord account (OAuth).

Instructions:
1. If the user wants to send a message, use send_message with the channel ID and content.
2. If the user wants to see channels in a server, use list_channels with the guild/server ID.
3. If the user wants to see what servers they are in, use list_servers.
4. If Discord is not yet connected, use connect_discord first.
5. If the user's request is ambiguous or missing required IDs, ask for clarification WITHOUT calling any tools.
6. After getting tool results, provide a clear summary to the user."""

    tools = (
        send_message,
        list_channels,
        list_servers,
        connect_discord,
    )
