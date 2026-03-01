"""Integration tests for ShippingAgent — tool selection, argument extraction, response quality.

ShippingAgent tools:
  track_shipment (with action parameter: query_one, query_all, update, delete, history)
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.lifestyle]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Track my package 1Z999AA10123456784", ["track_shipment"]),
    ("Where is my order with tracking number 123456789012?", ["track_shipment"]),
    ("Show me all my shipments", ["track_shipment"]),
    ("Track my FedEx delivery status", ["track_shipment"]),
    ("Delete tracking for 1Z999AA10123456784", ["track_shipment"]),
    ("Show my past deliveries", ["track_shipment"]),
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

async def test_query_one_extracts_tracking_number(orchestrator_factory):
    """track_shipment with a specific tracking number should use action=query_one."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Track package 1Z999AA10123456784")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "track_shipment"]
    assert calls, "Expected track_shipment to be called"

    args = calls[0]["arguments"]
    # Should extract the tracking number
    tracking = args.get("tracking_number", "") or args.get("query", "")
    assert "1Z999AA10123456784" in tracking or "1Z999AA" in str(args), (
        f"Expected tracking number in args, got {args}"
    )


async def test_query_all_action(orchestrator_factory):
    """Asking about all shipments should use action=query_all."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Show me all my packages")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "track_shipment"]
    assert calls, "Expected track_shipment to be called"

    args = calls[0]["arguments"]
    action = args.get("action", "")
    assert action == "query_all", f"Expected action=query_all, got {args}"


async def test_history_action(orchestrator_factory):
    """Asking about past deliveries should use action=history."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Show my delivery history")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "track_shipment"]
    assert calls, "Expected track_shipment to be called"

    args = calls[0]["arguments"]
    action = args.get("action", "")
    assert action == "history", f"Expected action=history, got {args}"


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_track_package(orchestrator_factory, llm_judge):
    """Tracking a package should present status and estimated delivery."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Where is my package 1Z999AA10123456784?"
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Where is my package 1Z999AA10123456784?",
        response=response,
        criteria=(
            "The response should include the shipment status (In Transit), "
            "carrier (UPS), and estimated delivery date. It should be clear "
            "and concise."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"


async def test_response_quality_all_shipments(orchestrator_factory, llm_judge):
    """Listing all shipments should present them clearly."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Show me all my shipments")
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Show me all my shipments",
        response=response,
        criteria=(
            "The response should present shipment or package information. "
            "It should mention tracking details, carrier, or status. "
            "It should not be an error message."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"
