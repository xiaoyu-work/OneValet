"""Integration tests for EmailAgent.

Tests tool selection, argument extraction, and response quality for:
- search_emails: Search emails across connected accounts
- send_email: Send a new email
- reply_email: Reply to an email by message_id
- delete_emails: Delete emails by message_ids
- archive_emails: Archive emails by message_ids
- mark_as_read: Mark emails as read by message_ids
"""

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Check my email", ["search_emails"]),
    ("Do I have any unread emails?", ["search_emails"]),
    ("Show emails from John", ["search_emails"]),
    ("Find the email about Q4 Report", ["search_emails"]),
    ("Send an email to alice@example.com about the meeting", ["send_email"]),
    ("Email bob@company.com saying I'll be late", ["send_email"]),
    ("Reply to the email from my boss saying sounds good", ["reply_email", "search_emails"]),
    ("Delete the promotional emails", ["delete_emails", "search_emails"]),
    ("Archive all emails from Amazon", ["archive_emails", "search_emails"]),
    ("Mark all emails as read", ["mark_as_read", "search_emails"]),
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

async def test_extracts_send_email_fields(orchestrator_factory):
    """send_email should receive to, subject, and body from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Send an email to alice@example.com with subject Project Update "
        "saying The project is on track for Friday delivery",
    )

    send_calls = [c for c in recorder.tool_calls if c["tool_name"] == "send_email"]
    assert send_calls, "send_email was never called"

    args = send_calls[0]["arguments"]
    assert "alice@example.com" in args.get("to", ""), (
        f"Expected to='alice@example.com', got '{args.get('to')}'"
    )
    assert args.get("body"), "body should not be empty"


async def test_extracts_search_sender_filter(orchestrator_factory):
    """search_emails should receive sender filter when user searches by sender."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Show me emails from boss@company.com")

    search_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "search_emails"
    ]
    assert search_calls, "search_emails was never called"

    args = search_calls[0]["arguments"]
    # The sender or query field should contain the email/name
    sender = args.get("sender", "") or ""
    query = args.get("query", "") or ""
    combined = f"{sender} {query}".lower()
    assert "boss" in combined or "boss@company.com" in combined, (
        f"Expected sender/query to reference 'boss@company.com', got sender='{sender}', query='{query}'"
    )


async def test_extracts_search_query_keywords(orchestrator_factory):
    """search_emails should receive keyword filter for subject-based searches."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Find the email about quarterly report")

    search_calls = [
        c for c in recorder.tool_calls if c["tool_name"] == "search_emails"
    ]
    assert search_calls, "search_emails was never called"

    args = search_calls[0]["arguments"]
    query = args.get("query", "").lower()
    assert "quarterly" in query or "report" in query, (
        f"Expected query containing 'quarterly' or 'report', got '{query}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_check_inbox(orchestrator_factory, llm_judge):
    """Checking emails should produce a readable listing of messages."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Check my inbox")

    passed = await llm_judge(
        "Check my inbox",
        result,
        "The response should present a list of emails with senders and subjects "
        "in a readable format. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result}"


async def test_response_quality_send(orchestrator_factory, llm_judge):
    """Sending an email should confirm the action with recipient details."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user",
        "Send an email to alice@example.com saying the meeting is confirmed",
    )

    passed = await llm_judge(
        "Send an email to alice@example.com saying the meeting is confirmed",
        result,
        "The response should confirm that an email was sent (or drafted) to "
        "alice@example.com. It should acknowledge the action positively.",
    )
    assert passed, f"LLM judge failed. Response: {result}"
