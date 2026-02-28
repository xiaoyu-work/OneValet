"""Integration tests for ExpenseAgent.

Tests tool selection, argument extraction, and response quality for:
- log_expense: Log a new expense with amount, category, optional details
- query_expenses: List expenses for a time period with optional filters
- delete_expense: Delete an expense by keyword search
- spending_summary: Show spending breakdown by category
- set_budget: Set a monthly spending limit
- budget_status: Show current budget utilization
- upload_receipt: Save a receipt image to storage
- search_receipts: Search saved receipts by keywords
"""

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("I spent $15 on lunch today", ["log_expense"]),
    ("Uber ride $12 yesterday", ["log_expense"]),
    ("Coffee at Starbucks $5.50", ["log_expense"]),
    ("Show me my expenses this month", ["query_expenses"]),
    ("How much did I spend last week?", ["query_expenses", "spending_summary"]),
    ("Delete the Starbucks expense from yesterday", ["delete_expense"]),
    ("Remove the $5 coffee charge", ["delete_expense"]),
    ("Give me a spending summary for February", ["spending_summary"]),
    ("Breakdown of my spending this month", ["spending_summary"]),
    ("Set my food budget to $500 per month", ["set_budget"]),
    ("How much budget do I have left?", ["budget_status"]),
    ("Find my receipt from the restaurant", ["search_receipts"]),
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

async def test_extracts_amount_and_category(orchestrator_factory):
    """log_expense should receive the correct amount and an appropriate category."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "I had lunch for $15 at Chipotle")

    log_calls = [c for c in recorder.tool_calls if c["tool_name"] == "log_expense"]
    assert log_calls, "log_expense was never called"

    args = log_calls[0]["arguments"]
    assert args.get("amount") == 15 or args.get("amount") == 15.0, (
        f"Expected amount=15, got {args.get('amount')}"
    )
    assert args.get("category", "").lower() == "food", (
        f"Expected category='food', got {args.get('category')}"
    )


async def test_extracts_merchant(orchestrator_factory):
    """log_expense should populate the merchant field when a business name is given."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Paid $8 at Starbucks for coffee")

    log_calls = [c for c in recorder.tool_calls if c["tool_name"] == "log_expense"]
    assert log_calls, "log_expense was never called"

    args = log_calls[0]["arguments"]
    merchant = args.get("merchant", "").lower()
    assert "starbucks" in merchant, (
        f"Expected merchant to contain 'starbucks', got '{merchant}'"
    )


async def test_extracts_query_period(orchestrator_factory):
    """query_expenses / spending_summary should receive the right period."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Show my spending for last month")

    relevant = [
        c for c in recorder.tool_calls
        if c["tool_name"] in ("query_expenses", "spending_summary")
    ]
    assert relevant, "Neither query_expenses nor spending_summary was called"

    args = relevant[0]["arguments"]
    period = args.get("period", "").lower()
    assert "last_month" in period or "last" in period, (
        f"Expected period containing 'last_month', got '{period}'"
    )


async def test_budget_amount_extraction(orchestrator_factory):
    """set_budget should receive the correct monthly_limit and category."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Set a $300 budget for transport")

    budget_calls = [c for c in recorder.tool_calls if c["tool_name"] == "set_budget"]
    assert budget_calls, "set_budget was never called"

    args = budget_calls[0]["arguments"]
    assert args.get("monthly_limit") == 300 or args.get("monthly_limit") == 300.0
    assert "transport" in args.get("category", "").lower()


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_log(orchestrator_factory, llm_judge):
    """After logging an expense the response should confirm the amount and category."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Lunch $15 at Chipotle")

    passed = await llm_judge(
        "Lunch $15 at Chipotle",
        result,
        "The response should confirm that an expense of approximately $15 in the food "
        "category was logged. It should mention the amount and ideally reference the "
        "merchant or description.",
    )
    assert passed, f"LLM judge failed. Response: {result}"


async def test_response_quality_query(orchestrator_factory, llm_judge):
    """Querying expenses should produce a readable summary."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Show my expenses this month")

    passed = await llm_judge(
        "Show my expenses this month",
        result,
        "The response should present expense data in a readable format, mentioning "
        "amounts and categories or merchants. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result}"
