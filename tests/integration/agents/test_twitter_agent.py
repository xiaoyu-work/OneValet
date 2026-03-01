"""Integration tests for TwitterComposioAgent.

Tests tool selection, argument extraction, and response quality for:
- post_tweet: Post a new tweet on Twitter/X
- get_timeline: Fetch recent tweets from home timeline
- search_tweets: Search for recent tweets matching a query
- lookup_user: Look up a Twitter/X user by username
- connect_twitter: Connect Twitter/X account via OAuth
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.communication]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Post a tweet saying 'Just launched our new product!'", ["post_tweet"]),
    ("Show me my Twitter timeline", ["get_timeline"]),
    ("Search Twitter for tweets about AI startups", ["search_tweets"]),
    ("Look up the Twitter user @elonmusk", ["lookup_user"]),
]


@pytest.mark.parametrize(
    "user_input,expected_tools",
    TOOL_SELECTION_CASES,
    ids=[c[0][:40] for c in TOOL_SELECTION_CASES],
)
async def test_tool_selection(orchestrator_factory, user_input, expected_tools):
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", user_input)
    tools_called = [c["tool_name"] for c in recorder.tool_calls]
    assert any(t in tools_called for t in expected_tools), (
        f"Expected one of {expected_tools}, got {tools_called}"
    )


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

async def test_extracts_tweet_text(orchestrator_factory):
    """post_tweet should receive the correct text content from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Tweet this: Excited to announce our Series A funding!",
    )

    tweet_calls = [c for c in recorder.tool_calls if c["tool_name"] == "post_tweet"]
    assert tweet_calls, "post_tweet was never called"

    args = tweet_calls[0]["arguments"]
    text = args.get("text", "").lower()
    assert "series a" in text or "funding" in text, (
        f"Expected tweet text to contain funding announcement, got '{args.get('text')}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_search(orchestrator_factory, llm_judge):
    """Searching tweets should produce a readable summary of matching tweets."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Search Twitter for tweets about machine learning"
    )

    passed = await llm_judge(
        "Search Twitter for tweets about machine learning",
        result.raw_message,
        "The response should present search results from Twitter, mentioning "
        "tweet content or authors. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result.raw_message}"
