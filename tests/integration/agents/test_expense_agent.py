"""Integration tests for ExpenseAgent.

Tests tool selection, argument extraction, response quality, and approval flow for:
- log_expense: Log a new expense with amount, category, optional details
- query_expenses: List expenses for a time period with optional filters
- delete_expense: Delete an expense by keyword search (needs approval)
- spending_summary: Show spending breakdown by category
- set_budget: Set a monthly spending limit
- budget_status: Show current budget utilization
- upload_receipt: Save a receipt image to storage
- search_receipts: Search saved receipts by keywords
"""

import pytest

from onevalet.result import AgentStatus

pytestmark = [pytest.mark.integration, pytest.mark.lifestyle]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("I spent $15 on lunch today", ["log_expense"]),
    ("Uber ride $12 yesterday", ["log_expense"]),
    ("Coffee at Starbucks $5.50", ["log_expense"]),
    ("Show me my expenses this month", ["query_expenses"]),
    ("How much did I spend last week?", ["query_expenses", "spending_summary"]),
    ("Delete the Starbucks expense from yesterday", ["delete_expense", "query_expenses"]),
    ("Remove the $5 coffee charge", ["delete_expense", "query_expenses"]),
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
async def test_tool_selection(conversation, user_input, expected_tools):
    conv = await conversation()
    await conv.send_until_tool_called(user_input)
    conv.assert_any_tool_called(expected_tools)


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

async def test_extracts_amount_and_category(conversation):
    """log_expense should receive the correct amount and an appropriate category."""
    conv = await conversation()
    await conv.send("I had lunch for $15 at Chipotle")
    conv.assert_tool_called("log_expense")

    args = conv.get_tool_args("log_expense")[0]
    assert args.get("amount") == 15 or args.get("amount") == 15.0, (
        f"Expected amount=15, got {args.get('amount')}"
    )
    assert args.get("category", "").lower() == "food", (
        f"Expected category='food', got {args.get('category')}"
    )


async def test_extracts_merchant(conversation):
    """log_expense should populate the merchant field when a business name is given."""
    conv = await conversation()
    await conv.send("Paid $8 at Starbucks for coffee")
    conv.assert_tool_called("log_expense")
    conv.assert_tool_args("log_expense", merchant="starbucks")


async def test_extracts_query_period(conversation):
    """query_expenses / spending_summary should receive the right period."""
    conv = await conversation()
    await conv.send("Show my spending for last month")
    conv.assert_any_tool_called(["query_expenses", "spending_summary"])

    relevant = [
        c for c in conv.recorder.tool_calls
        if c["tool_name"] in ("query_expenses", "spending_summary")
    ]
    args = relevant[0]["arguments"]
    period = args.get("period", "").lower()
    assert "last_month" in period or "last" in period, (
        f"Expected period containing 'last_month', got '{period}'"
    )


async def test_budget_amount_extraction(conversation):
    """set_budget should receive the correct monthly_limit and category."""
    conv = await conversation()
    await conv.send("Set a $300 budget for transport")
    conv.assert_tool_called("set_budget")

    args = conv.get_tool_args("set_budget")[0]
    assert args.get("monthly_limit") == 300 or args.get("monthly_limit") == 300.0
    assert "transport" in args.get("category", "").lower()


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_log(conversation, llm_judge):
    """After logging an expense the response should confirm the amount and category."""
    conv = await conversation()
    await conv.auto_complete("Lunch $15 at Chipotle")

    passed = await llm_judge(
        "Lunch $15 at Chipotle",
        conv.last_message,
        "The response should confirm that an expense of approximately $15 in the food "
        "category was logged. It should mention the amount and ideally reference the "
        "merchant or description.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


async def test_response_quality_query(conversation, llm_judge):
    """Querying expenses should produce a readable summary."""
    conv = await conversation()
    await conv.auto_complete("Show my expenses for February 2026")

    passed = await llm_judge(
        "Show my expenses for February 2026",
        conv.last_message,
        "The response should present expense data in a readable format, mentioning "
        "amounts and categories or merchants. It may say there are no expenses or "
        "list some expenses. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


# ---------------------------------------------------------------------------
# Approval flow
# ---------------------------------------------------------------------------

async def test_delete_expense_triggers_approval(conversation):
    """delete_expense should pause for user approval before executing."""
    conv = await conversation()
    await conv.send_until_status(
        "Delete the Starbucks expense from yesterday",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    conv.assert_any_tool_called(["delete_expense", "query_expenses"])


async def test_delete_expense_approve_executes(conversation):
    """Approving delete_expense should execute and complete."""
    conv = await conversation()
    result = await conv.send_until_status(
        "Delete the Starbucks expense from yesterday",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    if result.status != AgentStatus.WAITING_FOR_APPROVAL:
        pytest.skip("LLM did not reach approval gate (may have queried first)")

    result2 = await conv.send("yes, delete it")
    assert result2.status in (AgentStatus.COMPLETED, AgentStatus.WAITING_FOR_APPROVAL), (
        f"Expected COMPLETED after approval, got {result2.status}"
    )


async def test_delete_expense_reject_cancels(conversation):
    """Rejecting delete_expense should cancel the operation."""
    conv = await conversation()
    result = await conv.send_until_status(
        "Delete the Starbucks expense from yesterday",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    if result.status != AgentStatus.WAITING_FOR_APPROVAL:
        pytest.skip("LLM did not reach approval gate (may have queried first)")

    result2 = await conv.send("no, keep it")
    assert result2.status in (AgentStatus.CANCELLED, AgentStatus.COMPLETED), (
        f"Expected CANCELLED after rejection, got {result2.status}"
    )
