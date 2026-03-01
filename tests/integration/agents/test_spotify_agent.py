"""Integration tests for SpotifyComposioAgent.

Tests tool selection, argument extraction, and response quality for:
- play_music: Start or resume Spotify playback
- pause_music: Pause the current playback
- search_music: Search Spotify for tracks, albums, artists, or playlists
- get_playlists: List the current user's Spotify playlists
- add_to_playlist: Add items to a Spotify playlist
- now_playing: Get the currently playing track
- connect_spotify: Connect Spotify account via OAuth
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.lifestyle]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Play some jazz music on Spotify", ["play_music", "search_music"]),
    ("Pause the music on Spotify", ["pause_music"]),
    ("Search Spotify for Bohemian Rhapsody by Queen", ["search_music"]),
    ("Show me my Spotify playlists", ["get_playlists"]),
    ("What song is currently playing on Spotify?", ["now_playing"]),
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
    """search_music should receive the correct query and type."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Search Spotify for the album 'Abbey Road' by The Beatles",
    )

    search_calls = [c for c in recorder.tool_calls if c["tool_name"] == "search_music"]
    assert search_calls, "search_music was never called"

    args = search_calls[0]["arguments"]
    query = args.get("query", "").lower()
    assert "abbey road" in query or "beatles" in query, (
        f"Expected query to reference Abbey Road or Beatles, got '{args.get('query')}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_playlists(orchestrator_factory, llm_judge):
    """Getting playlists should produce a readable list of playlists."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Show me my Spotify playlists"
    )

    passed = await llm_judge(
        "Show me my Spotify playlists",
        result.raw_message,
        "The response should present a list of Spotify playlists, mentioning "
        "playlist names or track counts. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result.raw_message}"
