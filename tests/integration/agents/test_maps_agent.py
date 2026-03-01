"""Integration tests for MapsAgent.

Tests tool selection, argument extraction, and response quality for:
- search_places: Search for places, restaurants, businesses by query and location
- get_directions: Get driving/transit/walking directions between two locations
- check_air_quality: Check current AQI for a location
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.travel]


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
    ("Get directions from 100 Broadway to Whole Foods", ["get_directions"]),
    ("What's the air quality in Beijing?", ["check_air_quality"]),
    ("Is the air safe to breathe in LA today?", ["check_air_quality"]),
    ("Check AQI in San Francisco", ["check_air_quality"]),
]


@pytest.mark.parametrize(
    "user_input,expected_tools",
    TOOL_SELECTION_CASES,
    ids=[c[0][:40] for c in TOOL_SELECTION_CASES],
)
async def test_tool_selection(conversation, user_input, expected_tools):
    conv = await conversation()
    await conv.send_until_tool_called(user_input)
    conv.assert_any_tool_called(expected_tools)


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

async def test_extracts_search_query_and_location(conversation):
    """search_places should receive the query and location from the user message."""
    conv = await conversation()
    await conv.send("Find sushi restaurants in downtown Portland")
    conv.assert_tool_called("search_places")

    args = conv.get_tool_args("search_places")[0]
    query = args.get("query", "").lower()
    location = args.get("location", "").lower()

    assert "sushi" in query or "sushi" in location, (
        f"Expected query containing 'sushi', got query='{query}', location='{location}'"
    )
    assert "portland" in location or "portland" in query, (
        f"Expected location containing 'portland', got location='{location}', query='{query}'"
    )


async def test_extracts_directions_origin_and_destination(conversation):
    """get_directions should receive origin and destination from the user message."""
    conv = await conversation()
    await conv.send("Get me directions from Times Square to Central Park")
    conv.assert_tool_called("get_directions")

    args = conv.get_tool_args("get_directions")[0]
    origin = args.get("origin", "").lower()
    destination = args.get("destination", "").lower()

    assert "times square" in origin or "times" in origin, (
        f"Expected origin containing 'times square', got '{origin}'"
    )
    assert "central park" in destination or "central" in destination, (
        f"Expected destination containing 'central park', got '{destination}'"
    )


async def test_extracts_air_quality_location(conversation):
    """check_air_quality should receive the correct location."""
    conv = await conversation()
    await conv.send("What's the air quality in Tokyo?")
    conv.assert_tool_called("check_air_quality")
    conv.assert_tool_args("check_air_quality", location="tokyo")


async def test_extracts_directions_travel_mode(conversation):
    """get_directions should receive a walking mode when specified by the user."""
    conv = await conversation()
    await conv.send("Walking directions from Times Square to the Metropolitan Museum")
    conv.assert_tool_called("get_directions")

    args = conv.get_tool_args("get_directions")[0]
    mode = args.get("mode", "").lower()
    assert "walk" in mode, (
        f"Expected mode containing 'walk', got '{mode}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_search_places(conversation, llm_judge):
    """Searching for places should return a readable listing with details."""
    conv = await conversation()
    msg = "Find Italian restaurants near downtown Seattle"
    await conv.auto_complete(msg)

    passed = await llm_judge(
        msg,
        conv.last_message,
        "The response should list restaurant results with names and possibly "
        "addresses or ratings. It should be a helpful, readable list and not "
        "an error message.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


async def test_response_quality_directions(conversation, llm_judge):
    """Getting directions should return distance, duration, and steps."""
    conv = await conversation()
    msg = "How do I get from Union Square to Golden Gate Bridge?"
    await conv.auto_complete(msg)

    passed = await llm_judge(
        msg,
        conv.last_message,
        "The response should provide directions or route information. "
        "It should mention distance, travel time, or steps. "
        "It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


async def test_response_quality_air_quality(conversation, llm_judge):
    """Air quality check should return the AQI and a category/recommendation."""
    conv = await conversation()
    msg = "What's the air quality like in San Francisco?"
    await conv.auto_complete(msg)

    passed = await llm_judge(
        msg,
        conv.last_message,
        "The response should mention the air quality index (AQI) value and/or "
        "a category (Good, Moderate, etc.) for San Francisco. It should be "
        "informative and not an error.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"
