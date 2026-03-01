"""
TwitterComposioAgent - Agent for Twitter/X operations via Composio.

Provides post tweets, view timeline, search tweets, and look up users
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

# Composio action ID constants for Twitter/X
_ACTION_CREATE_POST = "TWITTER_CREATION_OF_A_POST"
_ACTION_HOME_TIMELINE = "TWITTER_USER_HOME_TIMELINE_BY_USER_ID"
_ACTION_RECENT_SEARCH = "TWITTER_RECENT_SEARCH"
_ACTION_USER_LOOKUP = "TWITTER_USER_LOOKUP_BY_USERNAME"
_APP_NAME = "twitter"


def _check_api_key() -> str | None:
    """Return error message if Composio API key is not configured, else None."""
    if not os.getenv("COMPOSIO_API_KEY"):
        return "Error: Composio API key not configured. Please add it in Settings."
    return None


# =============================================================================
# Approval preview functions
# =============================================================================

async def _post_tweet_preview(args: dict, context) -> str:
    text = args.get("text", "")
    preview = text[:100] + "..." if len(text) > 100 else text
    return f"Post tweet?\n\nText: {preview}"


# =============================================================================
# Tool executors
# =============================================================================

@tool(needs_approval=True, risk_level="write", get_preview=_post_tweet_preview)
async def post_tweet(
    text: Annotated[str, "The text content of the tweet to post"],
    *,
    context: AgentToolContext,
) -> str:
    """Post a new tweet on Twitter/X."""

    if not text:
        return "Error: text is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_CREATE_POST,
            params={"text": text},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Tweet posted successfully.\n\n{result}"
        return f"Failed to post tweet: {result}"
    except Exception as e:
        logger.error(f"Twitter post_tweet failed: {e}", exc_info=True)
        return f"Error posting tweet: {e}"


@tool
async def get_timeline(
    limit: Annotated[int, "Number of tweets to fetch from the timeline"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Fetch recent tweets from your home timeline on Twitter/X."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_HOME_TIMELINE,
            params={"max_results": limit},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Home timeline tweets:\n\n{result}"
        return f"Failed to fetch timeline: {result}"
    except Exception as e:
        logger.error(f"Twitter get_timeline failed: {e}", exc_info=True)
        return f"Error fetching Twitter timeline: {e}"


@tool
async def search_tweets(
    query: Annotated[str, "Search query string for finding tweets"],
    limit: Annotated[int, "Maximum number of tweets to return"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """Search for recent tweets matching a query on Twitter/X."""

    if not query:
        return "Error: query is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_RECENT_SEARCH,
            params={"query": query, "max_results": limit},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Tweets matching '{query}':\n\n{result}"
        return f"Failed to search tweets: {result}"
    except Exception as e:
        logger.error(f"Twitter search_tweets failed: {e}", exc_info=True)
        return f"Error searching tweets: {e}"


@tool
async def lookup_user(
    username: Annotated[str, "Twitter/X username to look up (without the @ symbol)"],
    *,
    context: AgentToolContext,
) -> str:
    """Look up a Twitter/X user by their username."""

    if not username:
        return "Error: username is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_USER_LOOKUP,
            params={"username": username},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"User @{username}:\n\n{result}"
        return f"Failed to look up user: {result}"
    except Exception as e:
        logger.error(f"Twitter lookup_user failed: {e}", exc_info=True)
        return f"Error looking up Twitter user: {e}"


@tool
async def connect_twitter(
    entity_id: Annotated[str, "Entity ID for multi-user setups"] = "default",
    *,
    context: AgentToolContext,
) -> str:
    """Connect your Twitter/X account via OAuth. Returns a URL to complete authorization."""

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
                    f"Twitter/X is already connected (account ID: {conn.get('id', 'unknown')}). "
                    f"You can use the other tools to interact with Twitter/X."
                )

        # Initiate new connection
        data = await client.initiate_connection(app_name=_APP_NAME, entity_id=entity_id)

        redirect = data.get("redirectUrl", data.get("redirect_url", ""))
        if redirect:
            return (
                f"To connect Twitter/X, please open this URL in your browser:\n\n"
                f"{redirect}\n\n"
                f"After completing the authorization, the connection will be active."
            )

        conn_id = data.get("id", data.get("connectedAccountId", ""))
        status = data.get("status", "")
        if status.upper() == "ACTIVE":
            return f"Successfully connected to Twitter/X. Connection ID: {conn_id}"
        return f"Connection initiated for Twitter/X. Status: {status}."
    except Exception as e:
        logger.error(f"Twitter connect failed: {e}", exc_info=True)
        return f"Error connecting to Twitter/X: {e}"


# =============================================================================
# Agent
# =============================================================================

@valet(domain="communication")
class TwitterComposioAgent(StandardAgent):
    """Post tweets, view timeline, search tweets, and look up users on Twitter/X.
    Use when the user mentions Twitter, X, tweets, or social media posting."""

    max_turns = 5
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a Twitter/X assistant with access to Twitter tools via Composio.

Available tools:
- post_tweet: Post a new tweet on Twitter/X.
- get_timeline: Fetch recent tweets from your home timeline.
- search_tweets: Search for recent tweets matching a query.
- lookup_user: Look up a Twitter/X user by their username.
- connect_twitter: Connect your Twitter/X account (OAuth).

Instructions:
1. If the user wants to post a tweet, use post_tweet with the text content.
2. If the user wants to see their timeline, use get_timeline with an optional limit.
3. If the user wants to search for tweets, use search_tweets with a query string.
4. If the user wants to find a user, use lookup_user with the username (without @).
5. If Twitter/X is not yet connected, use connect_twitter first.
6. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
7. After getting tool results, provide a clear summary to the user."""

    tools = (
        post_tweet,
        get_timeline,
        search_tweets,
        lookup_user,
        connect_twitter,
    )
