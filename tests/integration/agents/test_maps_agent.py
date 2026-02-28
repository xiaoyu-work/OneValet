"""Integration tests for MapsAgent.

Tests tool selection, argument extraction, and response quality for:
- search_places: Search for places, restaurants, businesses by query and location
- get_directions: Get driving/transit/walking directions between two locations
- check_air_quality: Check current AQI for a location
"""

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Find Italian restaurants near downtown Seattle", ["search_places"]),
    ("Where's the nearest gas station?", ["search_places"]),
    ("Coffee shops in San Francisco", ["search_places"]),
    ("Best pizza places in Brooklyn", ["search_places"]),
    ("How do I get to the airport from downtown?", ["get_directions"]),
    ("Directions from 123 Main St to Central Park", ["get_directions"]),
    ("Navigate to Whole Foods from my office", ["get_directions"]),
    ("What's the air quality in Beijing?", ["check_air_quality"]),
    ("Is the air safe to breathe in LA today?", ["check_air_quality"]),
    ("Check AQI in San Francisco", ["check_air_quality"]),
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

async def test_extracts_search_query_and_location(orchestrator_factory):
    """search_places should receive the query and location from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user", "Find sushi restaurants in downtown Portland"
    )

    search_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "search_places"
    ]
    assert search_calls, "search_places was never called"

    args = search_calls[0]["arguments"]
    query = args.get("query", "").lower()
    location = args.get("location", "").lower()

    assert "sushi" in query or "sushi" in location, (
        f"Expected query containing 'sushi', got query='{query}', location='{location}'"
    )
    assert "portland" in location or "portland" in query, (
        f"Expected location containing 'portland', got location='{location}', query='{query}'"
    )


async def test_extracts_directions_origin_and_destination(orchestrator_factory):
    """get_directions should receive origin and destination from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Get me directions from Times Square to Central Park",
    )

    dir_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "get_directions"
    ]
    assert dir_calls, "get_directions was never called"

    args = dir_calls[0]["arguments"]
    origin = args.get("origin", "").lower()
    destination = args.get("destination", "").lower()

    assert "times square" in origin or "times" in origin, (
        f"Expected origin containing 'times square', got '{origin}'"
    )
    assert "central park" in destination or "central" in destination, (
        f"Expected destination containing 'central park', got '{destination}'"
    )


async def test_extracts_air_quality_location(orchestrator_factory):
    """check_air_quality should receive the correct location."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "What's the air quality in Tokyo?")

    aqi_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "check_air_quality"
    ]
    assert aqi_calls, "check_air_quality was never called"

    args = aqi_calls[0]["arguments"]
    location = args.get("location", "").lower()
    assert "tokyo" in location, (
        f"Expected location containing 'tokyo', got '{location}'"
    )


async def test_extracts_directions_travel_mode(orchestrator_factory):
    """get_directions should receive a walking mode when specified by the user."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Walking directions from the hotel to the museum",
    )

    dir_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "get_directions"
    ]
    assert dir_calls, "get_directions was never called"

    args = dir_calls[0]["arguments"]
    mode = args.get("mode", "").lower()
    assert "walk" in mode, (
        f"Expected mode containing 'walk', got '{mode}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_search_places(orchestrator_factory, llm_judge):
    """Searching for places should return a readable listing with details."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Find Italian restaurants near downtown Seattle"
    )

    passed = await llm_judge(
        "Find Italian restaurants near downtown Seattle",
        result,
        "The response should list restaurant results with names and possibly "
        "addresses or ratings. It should be a helpful, readable list and not "
        "an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result}"


async def test_response_quality_directions(orchestrator_factory, llm_judge):
    """Getting directions should return distance, duration, and steps."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "How do I get from Union Square to Golden Gate Bridge?"
    )

    passed = await llm_judge(
        "How do I get from Union Square to Golden Gate Bridge?",
        result,
        "The response should provide directions with distance, travel time, "
        "and route steps or a summary. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result}"


async def test_response_quality_air_quality(orchestrator_factory, llm_judge):
    """Air quality check should return the AQI and a category/recommendation."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "What's the air quality like in San Francisco?"
    )

    passed = await llm_judge(
        "What's the air quality like in San Francisco?",
        result,
        "The response should mention the air quality index (AQI) value and/or "
        "a category (Good, Moderate, etc.) for San Francisco. It should be "
        "informative and not an error.",
    )
    assert passed, f"LLM judge failed. Response: {result}"
