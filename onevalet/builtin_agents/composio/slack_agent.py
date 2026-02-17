"""
SlackComposioAgent - Domain agent for Slack operations via Composio.

Provides send/fetch messages, list channels, find users, and create reminders
using the Composio OAuth proxy platform.
"""

import os
import logging
from typing import Annotated

from onevalet import valet
from onevalet.standard_agent import StandardAgent, AgentToolContext
from onevalet.tool_decorator import tool

from .client import ComposioClient

logger = logging.getLogger(__name__)

# Composio action ID constants for Slack
_ACTION_SEND_MESSAGE = "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL"
_ACTION_FETCH_MESSAGES = "SLACK_FETCH_CONVERSATION_HISTORY"
_ACTION_LIST_CHANNELS = "SLACK_LIST_ALL_CHANNELS"
_ACTION_FIND_USERS = "SLACK_FIND_USERS"
_ACTION_CREATE_REMINDER = "SLACK_CREATE_A_REMINDER"
_APP_NAME = "slack"


def _check_api_key() -> str | None:
    """Return error message if Composio API key is not configured, else None."""
    if not os.getenv("COMPOSIO_API_KEY"):
        return "Error: Composio API key not configured. Please add it in Settings."
    return None


# =============================================================================
# Approval preview functions
# =============================================================================

async def _send_message_preview(args: dict, context) -> str:
    channel = args.get("channel", "")
    text = args.get("text", "")
    preview = text[:100] + "..." if len(text) > 100 else text
    return f"Send Slack message?\n\nChannel: {channel}\nMessage: {preview}"


async def _create_reminder_preview(args: dict, context) -> str:
    text = args.get("text", "")
    time = args.get("time", "")
    return f"Create Slack reminder?\n\nReminder: {text}\nTime: {time}"


# =============================================================================
# Tool executors
# =============================================================================

@tool(needs_approval=True, risk_level="write", get_preview=_send_message_preview)
async def send_message(
    channel: Annotated[str, "Channel name (e.g. '#general') or channel/user ID"],
    text: Annotated[str, "Message content to send"],
    *,
    context: AgentToolContext,
) -> str:
    """Send a message to a Slack channel or user."""

    if not channel:
        return "Error: channel is required."
    if not text:
        return "Error: text is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SEND_MESSAGE,
            params={"channel": channel, "text": text},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Message sent to {channel}.\n\n{result}"
        return f"Failed to send message: {result}"
    except Exception as e:
        logger.error(f"Slack send_message failed: {e}", exc_info=True)
        return f"Error sending Slack message: {e}"


@tool
async def fetch_messages(
    channel: Annotated[str, "Channel name or ID to fetch messages from"],
    limit: Annotated[int, "Number of messages to fetch"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """Fetch recent messages from a Slack channel."""

    if not channel:
        return "Error: channel is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_FETCH_MESSAGES,
            params={"channel": channel, "limit": limit},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Messages from {channel}:\n\n{result}"
        return f"Failed to fetch messages: {result}"
    except Exception as e:
        logger.error(f"Slack fetch_messages failed: {e}", exc_info=True)
        return f"Error fetching Slack messages: {e}"


@tool
async def list_channels(
    limit: Annotated[int, "Maximum number of channels to return"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """List all available Slack channels in the workspace."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_CHANNELS,
            params={"limit": limit},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Slack channels:\n\n{result}"
        return f"Failed to list channels: {result}"
    except Exception as e:
        logger.error(f"Slack list_channels failed: {e}", exc_info=True)
        return f"Error listing Slack channels: {e}"


@tool
async def find_users(
    query: Annotated[str, "Search keyword (name or email)"],
    *,
    context: AgentToolContext,
) -> str:
    """Search for Slack users by name or email."""

    if not query:
        return "Error: query is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_FIND_USERS,
            params={"query": query},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Slack users matching '{query}':\n\n{result}"
        return f"Failed to find users: {result}"
    except Exception as e:
        logger.error(f"Slack find_users failed: {e}", exc_info=True)
        return f"Error searching Slack users: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_create_reminder_preview)
async def create_reminder(
    text: Annotated[str, "Reminder text (what to be reminded about)"],
    time: Annotated[str, "When to remind, e.g. 'in 30 minutes', 'tomorrow at 9am', or Unix timestamp"],
    *,
    context: AgentToolContext,
) -> str:
    """Create a Slack reminder for a specific time."""

    if not text:
        return "Error: text is required."
    if not time:
        return "Error: time is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_CREATE_REMINDER,
            params={"text": text, "time": time},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Reminder created: {text}\n\n{result}"
        return f"Failed to create reminder: {result}"
    except Exception as e:
        logger.error(f"Slack create_reminder failed: {e}", exc_info=True)
        return f"Error creating Slack reminder: {e}"


@tool
async def connect_slack(
    entity_id: Annotated[str, "Entity ID for multi-user setups"] = "default",
    *,
    context: AgentToolContext,
) -> str:
    """Connect your Slack account via OAuth. Returns a URL to complete authorization."""

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
                    f"Slack is already connected (account ID: {conn.get('id', 'unknown')}). "
                    f"You can use the other tools to interact with Slack."
                )

        # Initiate new connection
        data = await client.initiate_connection(app_name=_APP_NAME, entity_id=entity_id)

        redirect = data.get("redirectUrl", data.get("redirect_url", ""))
        if redirect:
            return (
                f"To connect Slack, please open this URL in your browser:\n\n"
                f"{redirect}\n\n"
                f"After completing the authorization, the connection will be active."
            )

        conn_id = data.get("id", data.get("connectedAccountId", ""))
        status = data.get("status", "")
        if status.upper() == "ACTIVE":
            return f"Successfully connected to Slack. Connection ID: {conn_id}"
        return f"Connection initiated for Slack. Status: {status}."
    except Exception as e:
        logger.error(f"Slack connect failed: {e}", exc_info=True)
        return f"Error connecting to Slack: {e}"


# =============================================================================
# Domain Agent
# =============================================================================

@valet(capabilities=["slack", "messaging"])
class SlackComposioAgent(StandardAgent):
    """Send messages, fetch conversations, list channels, find users, and create
    reminders in Slack. Use when the user mentions Slack, channels, or wants to
    send/read messages on Slack."""

    max_turns = 5
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a Slack assistant with access to Slack tools via Composio.

Available tools:
- send_message: Send a message to a Slack channel or user.
- fetch_messages: Fetch recent messages from a channel.
- list_channels: List all available Slack channels.
- find_users: Search for Slack users by name or email.
- create_reminder: Create a Slack reminder.
- connect_slack: Connect your Slack account (OAuth).

Instructions:
1. If the user wants to send a message, use send_message with the channel name/ID and text.
2. If the user wants to read messages, use fetch_messages with the channel name/ID.
3. If the user wants to know what channels exist, use list_channels.
4. If the user wants to find someone, use find_users with a search query.
5. If the user wants a reminder, use create_reminder with the text and time.
6. If Slack is not yet connected, use connect_slack first.
7. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
8. After getting tool results, provide a clear summary to the user."""

    tools = [
        send_message,
        fetch_messages,
        list_channels,
        find_users,
        create_reminder,
        connect_slack,
    ]
