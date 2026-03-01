"""
LinkedInComposioAgent - Agent for LinkedIn operations via Composio.

Provides create posts and view profile information
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

# Composio action ID constants for LinkedIn
_ACTION_CREATE_POST = "LINKEDIN_CREATE_LINKED_IN_POST"
_ACTION_GET_MY_INFO = "LINKEDIN_GET_MY_INFO"
_APP_NAME = "linkedin"


def _check_api_key() -> str | None:
    """Return error message if Composio API key is not configured, else None."""
    if not os.getenv("COMPOSIO_API_KEY"):
        return "Error: Composio API key not configured. Please add it in Settings."
    return None


# =============================================================================
# Approval preview functions
# =============================================================================

async def _create_post_preview(args: dict, context) -> str:
    text = args.get("text", "")
    visibility = args.get("visibility", "PUBLIC")
    preview = text[:100] + "..." if len(text) > 100 else text
    return f"Create LinkedIn post?\n\nVisibility: {visibility}\nContent: {preview}"


# =============================================================================
# Tool executors
# =============================================================================

@tool(needs_approval=True, risk_level="write", get_preview=_create_post_preview)
async def create_post(
    text: Annotated[str, "The content/text of the LinkedIn post"],
    visibility: Annotated[str, "Post visibility: 'PUBLIC' or 'CONNECTIONS'"] = "PUBLIC",
    *,
    context: AgentToolContext,
) -> str:
    """Create a new post on LinkedIn."""

    if not text:
        return "Error: text is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_CREATE_POST,
            params={"text": text, "visibility": visibility},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"LinkedIn post created.\n\n{result}"
        return f"Failed to create post: {result}"
    except Exception as e:
        logger.error(f"LinkedIn create_post failed: {e}", exc_info=True)
        return f"Error creating LinkedIn post: {e}"


@tool
async def get_my_profile(
    *,
    context: AgentToolContext,
) -> str:
    """Get your LinkedIn profile information (name, headline, etc.)."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_MY_INFO,
            params={},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"LinkedIn profile info:\n\n{result}"
        return f"Failed to get profile info: {result}"
    except Exception as e:
        logger.error(f"LinkedIn get_my_profile failed: {e}", exc_info=True)
        return f"Error getting LinkedIn profile: {e}"


@tool
async def connect_linkedin(
    entity_id: Annotated[str, "Entity ID for multi-user setups"] = "default",
    *,
    context: AgentToolContext,
) -> str:
    """Connect your LinkedIn account via OAuth. Returns a URL to complete authorization."""

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
                    f"LinkedIn is already connected (account ID: {conn.get('id', 'unknown')}). "
                    f"You can use the other tools to interact with LinkedIn."
                )

        # Initiate new connection
        data = await client.initiate_connection(app_name=_APP_NAME, entity_id=entity_id)

        redirect = data.get("redirectUrl", data.get("redirect_url", ""))
        if redirect:
            return (
                f"To connect LinkedIn, please open this URL in your browser:\n\n"
                f"{redirect}\n\n"
                f"After completing the authorization, the connection will be active."
            )

        conn_id = data.get("id", data.get("connectedAccountId", ""))
        status = data.get("status", "")
        if status.upper() == "ACTIVE":
            return f"Successfully connected to LinkedIn. Connection ID: {conn_id}"
        return f"Connection initiated for LinkedIn. Status: {status}."
    except Exception as e:
        logger.error(f"LinkedIn connect failed: {e}", exc_info=True)
        return f"Error connecting to LinkedIn: {e}"


# =============================================================================
# Agent
# =============================================================================

@valet(domain="communication")
class LinkedInComposioAgent(StandardAgent):
    """Create posts and view profile on LinkedIn. Use when the user mentions
    LinkedIn, professional networking, or wants to post on LinkedIn."""

    max_turns = 5
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a LinkedIn assistant with access to LinkedIn tools via Composio.

Available tools:
- create_post: Create a new post on LinkedIn with specified visibility.
- get_my_profile: Get your LinkedIn profile information (name, headline, etc.).
- connect_linkedin: Connect your LinkedIn account (OAuth).

Instructions:
1. If the user wants to create a post, use create_post with the text and optional visibility (PUBLIC or CONNECTIONS).
2. If the user wants to see their profile info, use get_my_profile.
3. If LinkedIn is not yet connected, use connect_linkedin first.
4. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
5. After getting tool results, provide a clear summary to the user."""

    tools = (
        create_post,
        get_my_profile,
        connect_linkedin,
    )
