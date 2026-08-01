"""
TwitterComposioAgent - Agent for Twitter/X operations via Composio.

Provides post tweets, delete tweets, view timeline, search tweets,
look up users, like/unlike, retweet, follow, followers/following,
bookmarks, direct messages, and user tweet history
using the Composio OAuth proxy platform.
"""

import logging
import os
from typing import Annotated, List, Optional

from koa import valet
from koa.models import AgentToolContext
from koa.standard_agent import StandardAgent
from koa.tool_decorator import tool

from .client import ComposioClient

logger = logging.getLogger(__name__)

# Composio action ID constants for Twitter/X
_ACTION_CREATE_POST = "TWITTER_CREATION_OF_A_POST"
_ACTION_DELETE_POST = "TWITTER_DELETION_OF_A_POST"
_ACTION_HOME_TIMELINE = "TWITTER_USER_HOME_TIMELINE_BY_USER_ID"
_ACTION_RECENT_SEARCH = "TWITTER_RECENT_SEARCH"
_ACTION_USER_LOOKUP = "TWITTER_USER_LOOKUP_BY_USERNAME"
_ACTION_LIKE_POST = "TWITTER_LIKES_A_POST"
_ACTION_UNLIKE_POST = "TWITTER_UNLIKE_A_POST"
_ACTION_RETWEET = "TWITTER_CREATION_OF_A_RETWEET"
_ACTION_FOLLOWERS = "TWITTER_FOLLOWERS_BY_USER_ID"
_ACTION_FOLLOWING = "TWITTER_FOLLOWING_BY_USER_ID"
_ACTION_FOLLOW = "TWITTER_FOLLOW_USER"
_ACTION_BOOKMARKS = "TWITTER_BOOKMARKS_BY_USER"
_ACTION_ADD_BOOKMARK = "TWITTER_ADD_POST_TO_BOOKMARKS"
_ACTION_SEND_DM = "TWITTER_CREATE_A_NEW_DM_CONVERSATION"
_ACTION_GET_DM_EVENTS = "TWITTER_GET_RECENT_DM_EVENTS"
_ACTION_USER_TWEETS = "TWITTER_USER_TWEETS"
_APP_NAME = "twitter"


def _check_api_key() -> Optional[str]:
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
            _ACTION_CREATE_POST, params={"text": text}, entity_id=context.tenant_id or "default"
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
            entity_id=context.tenant_id or "default",
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
            entity_id=context.tenant_id or "default",
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
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"User @{username}:\n\n{result}"
        return f"Failed to look up user: {result}"
    except Exception as e:
        logger.error(f"Twitter lookup_user failed: {e}", exc_info=True)
        return f"Error looking up Twitter user: {e}"


@tool(needs_approval=True, risk_level="write")
async def delete_post(
    tweet_id: Annotated[str, "The ID of the tweet to delete"],
    *,
    context: AgentToolContext,
) -> str:
    """Delete a tweet on Twitter/X."""

    if not tweet_id:
        return "Error: tweet_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_DELETE_POST,
            params={"tweet_id": tweet_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Tweet deleted successfully.\n\n{result}"
        return f"Failed to delete tweet: {result}"
    except Exception as e:
        logger.error(f"Twitter delete_post failed: {e}", exc_info=True)
        return f"Error deleting tweet: {e}"


@tool(needs_approval=True, risk_level="write")
async def like_post(
    user_id: Annotated[str, "The ID of the authenticated user"],
    tweet_id: Annotated[str, "The ID of the tweet to like"],
    *,
    context: AgentToolContext,
) -> str:
    """Like a tweet on Twitter/X."""

    if not user_id:
        return "Error: user_id is required."
    if not tweet_id:
        return "Error: tweet_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIKE_POST,
            params={"user_id": user_id, "tweet_id": tweet_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Tweet liked successfully.\n\n{result}"
        return f"Failed to like tweet: {result}"
    except Exception as e:
        logger.error(f"Twitter like_post failed: {e}", exc_info=True)
        return f"Error liking tweet: {e}"


@tool(needs_approval=True, risk_level="write")
async def unlike_post(
    user_id: Annotated[str, "The ID of the authenticated user"],
    tweet_id: Annotated[str, "The ID of the tweet to unlike"],
    *,
    context: AgentToolContext,
) -> str:
    """Unlike a tweet on Twitter/X."""

    if not user_id:
        return "Error: user_id is required."
    if not tweet_id:
        return "Error: tweet_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_UNLIKE_POST,
            params={"user_id": user_id, "tweet_id": tweet_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Tweet unliked successfully.\n\n{result}"
        return f"Failed to unlike tweet: {result}"
    except Exception as e:
        logger.error(f"Twitter unlike_post failed: {e}", exc_info=True)
        return f"Error unliking tweet: {e}"


@tool(needs_approval=True, risk_level="write")
async def retweet(
    user_id: Annotated[str, "The ID of the authenticated user"],
    tweet_id: Annotated[str, "The ID of the tweet to retweet"],
    *,
    context: AgentToolContext,
) -> str:
    """Retweet a tweet on Twitter/X."""

    if not user_id:
        return "Error: user_id is required."
    if not tweet_id:
        return "Error: tweet_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_RETWEET,
            params={"user_id": user_id, "tweet_id": tweet_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Retweeted successfully.\n\n{result}"
        return f"Failed to retweet: {result}"
    except Exception as e:
        logger.error(f"Twitter retweet failed: {e}", exc_info=True)
        return f"Error retweeting: {e}"


@tool
async def get_followers(
    user_id: Annotated[str, "The ID of the user whose followers to retrieve"],
    max_results: Annotated[int, "Maximum number of followers to return"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Get the list of followers for a Twitter/X user."""

    if not user_id:
        return "Error: user_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_FOLLOWERS,
            params={"user_id": user_id, "max_results": max_results},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Followers:\n\n{result}"
        return f"Failed to get followers: {result}"
    except Exception as e:
        logger.error(f"Twitter get_followers failed: {e}", exc_info=True)
        return f"Error getting followers: {e}"


@tool
async def get_following(
    user_id: Annotated[str, "The ID of the user whose following list to retrieve"],
    max_results: Annotated[int, "Maximum number of following users to return"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Get the list of users that a Twitter/X user is following."""

    if not user_id:
        return "Error: user_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_FOLLOWING,
            params={"user_id": user_id, "max_results": max_results},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Following:\n\n{result}"
        return f"Failed to get following list: {result}"
    except Exception as e:
        logger.error(f"Twitter get_following failed: {e}", exc_info=True)
        return f"Error getting following list: {e}"


@tool(needs_approval=True, risk_level="write")
async def follow_user(
    source_user_id: Annotated[str, "The ID of the authenticated user who will follow"],
    target_user_id: Annotated[str, "The ID of the user to follow"],
    *,
    context: AgentToolContext,
) -> str:
    """Follow a user on Twitter/X."""

    if not source_user_id:
        return "Error: source_user_id is required."
    if not target_user_id:
        return "Error: target_user_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_FOLLOW,
            params={"source_user_id": source_user_id, "target_user_id": target_user_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Followed user successfully.\n\n{result}"
        return f"Failed to follow user: {result}"
    except Exception as e:
        logger.error(f"Twitter follow_user failed: {e}", exc_info=True)
        return f"Error following user: {e}"


@tool
async def get_bookmarks(
    user_id: Annotated[str, "The ID of the authenticated user"],
    max_results: Annotated[int, "Maximum number of bookmarks to return"] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Get bookmarked tweets for a Twitter/X user."""

    if not user_id:
        return "Error: user_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_BOOKMARKS,
            params={"user_id": user_id, "max_results": max_results},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Bookmarked tweets:\n\n{result}"
        return f"Failed to get bookmarks: {result}"
    except Exception as e:
        logger.error(f"Twitter get_bookmarks failed: {e}", exc_info=True)
        return f"Error getting bookmarks: {e}"


@tool(needs_approval=True, risk_level="write")
async def add_bookmark(
    user_id: Annotated[str, "The ID of the authenticated user"],
    tweet_id: Annotated[str, "The ID of the tweet to bookmark"],
    *,
    context: AgentToolContext,
) -> str:
    """Bookmark a tweet on Twitter/X."""

    if not user_id:
        return "Error: user_id is required."
    if not tweet_id:
        return "Error: tweet_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_ADD_BOOKMARK,
            params={"user_id": user_id, "tweet_id": tweet_id},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Tweet bookmarked successfully.\n\n{result}"
        return f"Failed to bookmark tweet: {result}"
    except Exception as e:
        logger.error(f"Twitter add_bookmark failed: {e}", exc_info=True)
        return f"Error bookmarking tweet: {e}"


@tool(needs_approval=True, risk_level="write")
async def send_dm(
    participant_ids: Annotated[List[str], "List of user IDs to include in the DM conversation"],
    text: Annotated[str, "The text content of the direct message"],
    *,
    context: AgentToolContext,
) -> str:
    """Send a direct message on Twitter/X."""

    if not participant_ids:
        return "Error: participant_ids is required."
    if not text:
        return "Error: text is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SEND_DM,
            params={"participant_ids": participant_ids, "text": text},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"DM sent successfully.\n\n{result}"
        return f"Failed to send DM: {result}"
    except Exception as e:
        logger.error(f"Twitter send_dm failed: {e}", exc_info=True)
        return f"Error sending DM: {e}"


@tool
async def get_recent_dms(
    *,
    context: AgentToolContext,
) -> str:
    """Get recent direct message events on Twitter/X."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_DM_EVENTS,
            params={},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Recent DM events:\n\n{result}"
        return f"Failed to get DM events: {result}"
    except Exception as e:
        logger.error(f"Twitter get_recent_dms failed: {e}", exc_info=True)
        return f"Error getting DM events: {e}"


@tool
async def get_user_tweets(
    user_id: Annotated[str, "The ID of the user whose tweets to retrieve"],
    max_results: Annotated[int, "Maximum number of tweets to return"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """Get recent tweets posted by a specific Twitter/X user."""

    if not user_id:
        return "Error: user_id is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_USER_TWEETS,
            params={"user_id": user_id, "max_results": max_results},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"User tweets:\n\n{result}"
        return f"Failed to get user tweets: {result}"
    except Exception as e:
        logger.error(f"Twitter get_user_tweets failed: {e}", exc_info=True)
        return f"Error getting user tweets: {e}"


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
    """Post, delete, like, unlike, retweet, search, and manage tweets, followers,
    bookmarks, and direct messages on Twitter/X. Use when the user mentions
    Twitter, X, tweets, or social media posting."""

    max_turns = 15
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a Twitter/X assistant with access to Twitter tools via Composio.

Available tools:
- post_tweet: Post a new tweet on Twitter/X.
- delete_post: Delete a tweet by its ID.
- get_timeline: Fetch recent tweets from your home timeline.
- search_tweets: Search for recent tweets matching a query.
- lookup_user: Look up a Twitter/X user by their username.
- like_post: Like a tweet (requires user_id and tweet_id).
- unlike_post: Unlike a previously liked tweet.
- retweet: Retweet a tweet (requires user_id and tweet_id).
- get_followers: Get the list of followers for a user.
- get_following: Get the list of users a user is following.
- follow_user: Follow another user (requires source and target user IDs).
- get_bookmarks: Get bookmarked tweets for a user.
- add_bookmark: Bookmark a tweet.
- send_dm: Send a direct message to one or more users.
- get_recent_dms: Get recent direct message events.
- get_user_tweets: Get recent tweets posted by a specific user.
- connect_twitter: Connect your Twitter/X account (OAuth).

Instructions:
1. If the user wants to post a tweet, use post_tweet with the text content.
2. If the user wants to delete a tweet, use delete_post with the tweet ID.
3. If the user wants to see their timeline, use get_timeline with an optional limit.
4. If the user wants to search for tweets, use search_tweets with a query string.
5. If the user wants to find a user, use lookup_user with the username (without @).
6. If the user wants to like or unlike a tweet, use like_post or unlike_post.
7. If the user wants to retweet, use retweet with the user_id and tweet_id.
8. If the user wants to see followers or following, use get_followers or get_following.
9. If the user wants to follow someone, use follow_user with both user IDs.
10. If the user wants to manage bookmarks, use get_bookmarks or add_bookmark.
11. If the user wants to send or view DMs, use send_dm or get_recent_dms.
12. If the user wants to see a user's tweets, use get_user_tweets with the user_id.
13. If Twitter/X is not yet connected, use connect_twitter first.
14. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
15. After getting tool results, provide a clear summary to the user."""

    tools = (
        post_tweet,
        delete_post,
        get_timeline,
        search_tweets,
        lookup_user,
        like_post,
        unlike_post,
        retweet,
        get_followers,
        get_following,
        follow_user,
        get_bookmarks,
        add_bookmark,
        send_dm,
        get_recent_dms,
        get_user_tweets,
        connect_twitter,
    )
