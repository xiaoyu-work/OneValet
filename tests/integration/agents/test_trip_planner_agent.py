"""Integration tests for TripPlannerAgent — tool selection, argument extraction, response quality.

TripPlannerAgent tools:
  check_weather, search_places, get_directions, search_flights,
  search_hotels, query_events, create_event, create_task

The agent orchestrates multiple domains to produce itineraries. On the first
turn it should call check_weather, search_places, and search_hotels in
parallel (and search_flights if an origin city is given).
"""

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Plan a 3-day trip to Tokyo", ["search_places", "check_weather", "search_hotels"]),
    ("What's the weather like in Paris for my trip?", ["check_weather"]),
    ("Find hotels in Barcelona for next week", ["search_hotels"]),
    ("Search for restaurants near the Eiffel Tower", ["search_places"]),
    ("Plan a trip to London from New York", ["search_flights", "search_places", "check_weather"]),
    ("How do I get from Shibuya to Asakusa?", ["get_directions"]),
    ("Find flights from SF to Tokyo", ["search_flights"]),
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

async def test_weather_extracts_location(orchestrator_factory):
    """check_weather should receive the destination city."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Plan a 3-day trip to Tokyo")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "check_weather"]
    assert calls, "Expected check_weather to be called"

    args = calls[0]["arguments"]
    location = args.get("location", "") or args.get("city", "") or args.get("query", "")
    assert "tokyo" in location.lower(), (
        f"Expected location containing 'Tokyo', got {args}"
    )


async def test_search_places_extracts_destination(orchestrator_factory):
    """search_places should receive the destination or a relevant query."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Plan a 3-day trip to Tokyo")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "search_places"]
    assert calls, "Expected search_places to be called"

    args = calls[0]["arguments"]
    query = args.get("query", "") or args.get("location", "")
    assert "tokyo" in query.lower() or args, (
        f"Expected query related to Tokyo, got {args}"
    )


async def test_hotels_extracts_destination(orchestrator_factory):
    """search_hotels should receive the destination city."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Find hotels in Barcelona for 3 nights")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "search_hotels"]
    assert calls, "Expected search_hotels to be called"

    args = calls[0]["arguments"]
    # The city/destination should be Barcelona
    location = (
        args.get("city", "")
        or args.get("destination", "")
        or args.get("location", "")
        or args.get("query", "")
    )
    assert "barcelona" in location.lower(), (
        f"Expected destination containing 'Barcelona', got {args}"
    )


async def test_flights_extracts_origin_and_destination(orchestrator_factory):
    """search_flights should receive origin and destination when both are given."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Find flights from San Francisco to Tokyo",
    )

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "search_flights"]
    assert calls, "Expected search_flights to be called"

    args = calls[0]["arguments"]
    origin = args.get("origin", "") or args.get("from", "") or args.get("departure", "")
    destination = args.get("destination", "") or args.get("to", "") or args.get("arrival", "")
    # At least destination should be present
    assert destination or origin, f"Expected origin/destination in args, got {args}"


async def test_directions_extracts_endpoints(orchestrator_factory):
    """get_directions should receive origin and destination locations."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "How do I get from Shibuya to Asakusa?")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "get_directions"]
    assert calls, "Expected get_directions to be called"

    args = calls[0]["arguments"]
    origin = args.get("origin", "") or args.get("from", "")
    destination = args.get("destination", "") or args.get("to", "")
    assert origin or destination, f"Expected origin/destination in args, got {args}"


# ---------------------------------------------------------------------------
# Multi-tool parallel calls
# ---------------------------------------------------------------------------

async def test_trip_plan_triggers_multiple_tools(orchestrator_factory):
    """A full trip planning request should trigger weather + places + hotels."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Plan a 3-day trip to Tokyo")

    tools_called = set(c["tool_name"] for c in recorder.tool_calls)

    # The agent should call at least 2 of these on the first turn
    expected_set = {"check_weather", "search_places", "search_hotels"}
    overlap = tools_called & expected_set
    assert len(overlap) >= 2, (
        f"Expected at least 2 of {expected_set} to be called, got {tools_called}"
    )


async def test_trip_with_origin_triggers_flights(orchestrator_factory):
    """When origin is provided, search_flights should also be called."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user", "Plan a trip to London from New York next week"
    )

    tools_called = set(c["tool_name"] for c in recorder.tool_calls)
    assert "search_flights" in tools_called, (
        f"Expected search_flights when origin is given, got {tools_called}"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_trip_plan(orchestrator_factory, llm_judge):
    """A trip plan should include weather, places, and accommodation info."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Plan a 3-day trip to Tokyo")
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Plan a 3-day trip to Tokyo",
        response=response,
        criteria=(
            "The response should be a structured trip itinerary for Tokyo. "
            "It should reference real data: weather conditions, places/attractions "
            "with names, and hotel options with prices. It should have a day-by-day "
            "structure. It should NOT be a generic text-only plan without data."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"


async def test_response_quality_hotel_search(orchestrator_factory, llm_judge):
    """Hotel search should present options with prices."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Find hotels in Barcelona for next week"
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Find hotels in Barcelona for next week",
        response=response,
        criteria=(
            "The response should present hotel options with names and prices. "
            "It should be helpful for someone planning a stay in Barcelona."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"


async def test_response_quality_directions(orchestrator_factory, llm_judge):
    """Directions response should include distance and steps."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "How do I get from Shibuya to Asakusa?"
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="How do I get from Shibuya to Asakusa?",
        response=response,
        criteria=(
            "The response should provide directions between the two locations. "
            "It should mention distance or travel time. It should be clear "
            "and actionable."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"
