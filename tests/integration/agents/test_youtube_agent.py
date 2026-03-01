"""Integration tests for YouTubeComposioAgent.

Tests tool selection, argument extraction, and response quality for:
- search_videos: Search YouTube for videos matching a query
- get_video_details: Get detailed information about a specific video
- list_playlists: List playlists for the connected YouTube account
- connect_youtube: Connect YouTube account via OAuth
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.lifestyle]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Search YouTube for Python tutorial videos", ["search_videos"]),
    ("Find me some cooking recipe videos on YouTube", ["search_videos"]),
    ("Get details about YouTube video dQw4w9WgXcQ", ["get_video_details"]),
    ("Show me my YouTube playlists", ["list_playlists"]),
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

async def test_extracts_search_query(orchestrator_factory):
    """search_videos should receive the correct query from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Search YouTube for 'machine learning crash course' videos",
    )

    search_calls = [c for c in recorder.tool_calls if c["tool_name"] == "search_videos"]
    assert search_calls, "search_videos was never called"

    args = search_calls[0]["arguments"]
    query = args.get("query", "").lower()
    assert "machine learning" in query or "crash course" in query, (
        f"Expected query to reference machine learning, got '{args.get('query')}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_search(orchestrator_factory, llm_judge):
    """Searching videos should produce a readable list of results."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Search YouTube for sunset timelapse videos"
    )

    passed = await llm_judge(
        "Search YouTube for sunset timelapse videos",
        result.raw_message,
        "The response should present YouTube video search results, mentioning "
        "video titles, channels, or URLs. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result.raw_message}"
