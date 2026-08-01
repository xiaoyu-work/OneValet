"""
SlackComposioAgent - Agent for Slack operations via Composio.

Provides send/fetch messages, list/find/create/archive channels, find users,
manage reactions, user presence, invitations, status, and create reminders
using the Composio OAuth proxy platform.
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

# Composio action ID constants for Slack
_ACTION_SEND_MESSAGE = "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL"
_ACTION_FETCH_MESSAGES = "SLACK_FETCH_CONVERSATION_HISTORY"
_ACTION_LIST_CHANNELS = "SLACK_LIST_ALL_CHANNELS"
_ACTION_FIND_USERS = "SLACK_FIND_USERS"
_ACTION_CREATE_REMINDER = "SLACK_CREATE_A_REMINDER"
_ACTION_REPLY_MESSAGE = "SLACK_FETCH_MESSAGE_THREAD_FROM_A_CONVERSATION"
_ACTION_DELETE_MESSAGE = "SLACK_DELETES_A_MESSAGE_FROM_A_CHAT"
_ACTION_CREATE_CHANNEL = "SLACK_CREATE_CHANNEL"
_ACTION_ARCHIVE_CHANNEL = "SLACK_ARCHIVE_A_SLACK_CONVERSATION"
_ACTION_FIND_CHANNELS = "SLACK_FIND_CHANNELS"
_ACTION_FIND_USER_BY_EMAIL = "SLACK_FIND_USER_BY_EMAIL_ADDRESS"
_ACTION_ADD_REACTION = "SLACK_ADD_REACTION_TO_AN_ITEM"
_ACTION_GET_PRESENCE = "SLACK_GET_USER_PRESENCE_INFO"
_ACTION_INVITE_TO_CHANNEL = "SLACK_INVITE_USERS_TO_A_SLACK_CHANNEL"
_ACTION_SET_STATUS = "SLACK_SET_USER_CUSTOM_STATUS"
_APP_NAME = "slack"


def _check_api_key() -> Optional[str]:
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


async def _delete_message_preview(args: dict, context) -> str:
    channel = args.get("channel", "")
    ts = args.get("ts", "")
    return f"Delete Slack message?\n\nChannel: {channel}\nTimestamp: {ts}"


async def _create_channel_preview(args: dict, context) -> str:
    name = args.get("name", "")
    is_private = args.get("is_private", False)
    visibility = "private" if is_private else "public"
    return f"Create Slack channel?\n\nName: {name}\nVisibility: {visibility}"


async def _archive_channel_preview(args: dict, context) -> str:
    channel = args.get("channel", "")
    return f"Archive Slack channel?\n\nChannel: {channel}"


async def _invite_to_channel_preview(args: dict, context) -> str:
    channel = args.get("channel", "")
    users = args.get("users", "")
    return f"Invite users to Slack channel?\n\nChannel: {channel}\nUsers: {users}"


async def _set_status_preview(args: dict, context) -> str:
    text = args.get("status_text", "")
    emoji = args.get("status_emoji", "")
    return f"Set Slack status?\n\nText: {text}\nEmoji: {emoji}"


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
            entity_id=context.tenant_id or "default",
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
            entity_id=context.tenant_id or "default",
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
            _ACTION_LIST_CHANNELS, params={"limit": limit}, entity_id=context.tenant_id or "default"
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
            _ACTION_FIND_USERS, params={"query": query}, entity_id=context.tenant_id or "default"
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
    time: Annotated[
        str, "When to remind, e.g. 'in 30 minutes', 'tomorrow at 9am', or Unix timestamp"
    ],
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
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Reminder created: {text}\n\n{result}"
        return f"Failed to create reminder: {result}"
    except Exception as e:
        logger.error(f"Slack create_reminder failed: {e}", exc_info=True)
        return f"Error creating Slack reminder: {e}"


@tool
async def fetch_thread(
    channel: Annotated[str, "Channel name or ID containing the thread"],
    thread_ts: Annotated[str, "Timestamp of the parent message (thread root)"],
    limit: Annotated[int, "Maximum number of replies to fetch"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Fetch replies in a Slack message thread."""

    if not channel:
        return "Error: channel is required."
    if not thread_ts:
        return "Error: thread_ts is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_REPLY_MESSAGE,
            params={"channel": channel, "ts": thread_ts, "limit": limit},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Thread replies from {channel}:\n\n{result}"
        return f"Failed to fetch thread: {result}"
    except Exception as e:
        logger.error(f"Slack fetch_thread failed: {e}", exc_info=True)
        return f"Error fetching Slack thread: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_delete_message_preview)
async def delete_message(
    channel: Annotated[str, "Channel name or ID containing the message"],
    ts: Annotated[str, "Timestamp of the message to delete"],
    *,
    context: AgentToolContext,
) -> str:
    """Delete a message from a Slack channel."""

    if not channel:
        return "Error: channel is required."
    if not ts:
        return "Error: ts is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_DELETE_MESSAGE,
            params={"channel": channel, "ts": ts},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Message deleted from {channel}.\n\n{result}"
        return f"Failed to delete message: {result}"
    except Exception as e:
        logger.error(f"Slack delete_message failed: {e}", exc_info=True)
        return f"Error deleting Slack message: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_create_channel_preview)
async def create_channel(
    name: Annotated[str, "Channel name (lowercase, no spaces, use hyphens)"],
    is_private: Annotated[bool, "Whether the channel should be private"] = False,
    *,
    context: AgentToolContext,
) -> str:
    """Create a new Slack channel."""

    if not name:
        return "Error: name is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_CREATE_CHANNEL,
            params={"name": name, "is_private": is_private},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Channel '{name}' created.\n\n{result}"
        return f"Failed to create channel: {result}"
    except Exception as e:
        logger.error(f"Slack create_channel failed: {e}", exc_info=True)
        return f"Error creating Slack channel: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_archive_channel_preview)
async def archive_channel(
    channel: Annotated[str, "Channel name or ID to archive"],
    *,
    context: AgentToolContext,
) -> str:
    """Archive a Slack channel."""

    if not channel:
        return "Error: channel is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_ARCHIVE_CHANNEL,
            params={"channel": channel},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Channel '{channel}' archived.\n\n{result}"
        return f"Failed to archive channel: {result}"
    except Exception as e:
        logger.error(f"Slack archive_channel failed: {e}", exc_info=True)
        return f"Error archiving Slack channel: {e}"


@tool
async def find_channels(
    query: Annotated[str, "Search keyword to match channel name or topic"],
    *,
    context: AgentToolContext,
) -> str:
    """Search for Slack channels by name or topic."""

    if not query:
        return "Error: query is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_FIND_CHANNELS,
            params={"query": query},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Channels matching '{query}':\n\n{result}"
        return f"Failed to find channels: {result}"
    except Exception as e:
        logger.error(f"Slack find_channels failed: {e}", exc_info=True)
        return f"Error searching Slack channels: {e}"


@tool
async def find_user_by_email(
    email: Annotated[str, "Email address of the user to look up"],
    *,
    context: AgentToolContext,
) -> str:
    """Look up a Slack user by their email address."""

    if not email:
        return "Error: email is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_FIND_USER_BY_EMAIL,
            params={"email": email},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"User found for '{email}':\n\n{result}"
        return f"Failed to find user: {result}"
    except Exception as e:
        logger.error(f"Slack find_user_by_email failed: {e}", exc_info=True)
        return f"Error looking up Slack user by email: {e}"


@tool(needs_approval=True, risk_level="write")
async def add_reaction(
    channel: Annotated[str, "Channel name or ID containing the message"],
    timestamp: Annotated[str, "Timestamp of the message to react to"],
    name: Annotated[str, "Emoji name without colons (e.g. 'thumbsup', 'rocket')"],
    *,
    context: AgentToolContext,
) -> str:
    """Add an emoji reaction to a Slack message."""

    if not channel:
        return "Error: channel is required."
    if not timestamp:
        return "Error: timestamp is required."
    if not name:
        return "Error: name is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_ADD_REACTION,
            params={"channel": channel, "timestamp": timestamp, "name": name},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Reaction :{name}: added.\n\n{result}"
        return f"Failed to add reaction: {result}"
    except Exception as e:
        logger.error(f"Slack add_reaction failed: {e}", exc_info=True)
        return f"Error adding Slack reaction: {e}"


@tool
async def get_user_presence(
    user: Annotated[str, "User ID to check presence for"],
    *,
    context: AgentToolContext,
) -> str:
    """Check if a Slack user is currently active or away."""

    if not user:
        return "Error: user is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_PRESENCE,
            params={"user": user},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Presence info for {user}:\n\n{result}"
        return f"Failed to get presence: {result}"
    except Exception as e:
        logger.error(f"Slack get_user_presence failed: {e}", exc_info=True)
        return f"Error checking Slack user presence: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_invite_to_channel_preview)
async def invite_to_channel(
    channel: Annotated[str, "Channel name or ID to invite users to"],
    users: Annotated[str, "Comma-separated list of user IDs to invite"],
    *,
    context: AgentToolContext,
) -> str:
    """Invite users to a Slack channel."""

    if not channel:
        return "Error: channel is required."
    if not users:
        return "Error: users is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_INVITE_TO_CHANNEL,
            params={"channel": channel, "users": users},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Users invited to {channel}.\n\n{result}"
        return f"Failed to invite users: {result}"
    except Exception as e:
        logger.error(f"Slack invite_to_channel failed: {e}", exc_info=True)
        return f"Error inviting users to Slack channel: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_set_status_preview)
async def set_status(
    status_text: Annotated[str, "Status text to display (e.g. 'In a meeting')"],
    status_emoji: Annotated[str, "Status emoji (e.g. ':calendar:')"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Set the current user's custom Slack status."""

    if not status_text:
        return "Error: status_text is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        params = {"status_text": status_text}
        if status_emoji:
            params["status_emoji"] = status_emoji
        data = await client.execute_action(
            _ACTION_SET_STATUS,
            params=params,
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Status set to '{status_text}'.\n\n{result}"
        return f"Failed to set status: {result}"
    except Exception as e:
        logger.error(f"Slack set_status failed: {e}", exc_info=True)
        return f"Error setting Slack status: {e}"


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
# Agent
# =============================================================================


@valet(domain="communication")
class SlackComposioAgent(StandardAgent):
    """Send messages, fetch conversations, manage channels, find users, handle
    reactions, check presence, invite members, set status, and create reminders
    in Slack. Use when the user mentions Slack, channels, or wants to
    send/read messages on Slack."""

    max_turns = 15
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a Slack assistant with access to Slack tools via Composio.

Available tools:
- send_message: Send a message to a Slack channel or user.
- fetch_messages: Fetch recent messages from a channel.
- fetch_thread: Fetch replies in a message thread.
- delete_message: Delete a message from a channel.
- list_channels: List all available Slack channels.
- find_channels: Search for channels by name or topic.
- create_channel: Create a new Slack channel (public or private).
- archive_channel: Archive an existing Slack channel.
- find_users: Search for Slack users by name or email.
- find_user_by_email: Look up a Slack user by their email address.
- add_reaction: Add an emoji reaction to a message.
- get_user_presence: Check if a user is currently active or away.
- invite_to_channel: Invite users to a Slack channel.
- set_status: Set the current user's custom status and emoji.
- create_reminder: Create a Slack reminder.
- connect_slack: Connect your Slack account (OAuth).

Instructions:
1. If the user wants to send a message, use send_message with the channel name/ID and text.
2. If the user wants to read messages, use fetch_messages with the channel name/ID.
3. If the user wants to see thread replies, use fetch_thread with the channel and thread timestamp.
4. If the user wants to delete a message, use delete_message with the channel and message timestamp.
5. If the user wants to know what channels exist, use list_channels.
6. If the user wants to search for specific channels, use find_channels with a query.
7. If the user wants to create a channel, use create_channel with a name.
8. If the user wants to archive a channel, use archive_channel with the channel name/ID.
9. If the user wants to find someone, use find_users with a search query.
10. If the user has an email and wants the Slack user, use find_user_by_email.
11. If the user wants to react to a message, use add_reaction with the emoji name.
12. If the user wants to check someone's availability, use get_user_presence.
13. If the user wants to invite people to a channel, use invite_to_channel.
14. If the user wants to update their status, use set_status.
15. If the user wants a reminder, use create_reminder with the text and time.
16. If Slack is not yet connected, use connect_slack first.
17. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
18. After getting tool results, provide a clear summary to the user."""

    tools = (
        send_message,
        fetch_messages,
        fetch_thread,
        delete_message,
        list_channels,
        find_channels,
        create_channel,
        archive_channel,
        find_users,
        find_user_by_email,
        add_reaction,
        get_user_presence,
        invite_to_channel,
        set_status,
        create_reminder,
        connect_slack,
    )
