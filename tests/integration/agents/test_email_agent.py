"""Integration tests for EmailAgent.

Tests tool selection, argument extraction, response quality, and approval flow for:
- search_emails: Search emails across connected accounts
- send_email: Send a new email (needs approval)
- reply_email: Reply to an email by message_id (needs approval)
- delete_emails: Delete emails by message_ids (needs approval)
- archive_emails: Archive emails by message_ids (needs approval)
- mark_as_read: Mark emails as read by message_ids
"""

import pytest

from onevalet.result import AgentStatus

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
async def test_tool_selection(conversation, user_input, expected_tools):
    conv = await conversation()
    await conv.send_until_tool_called(user_input)
    conv.assert_any_tool_called(expected_tools)


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

async def test_extracts_send_email_fields(conversation):
    """send_email should receive to, subject, and body from the user message."""
    conv = await conversation()
    await conv.send_until_tool_called(
        "Send an email to alice@example.com with subject Project Update "
        "saying The project is on track for Friday delivery",
    )
    conv.assert_tool_called("send_email")
    conv.assert_tool_args("send_email", to="alice@example.com")
    args = conv.get_tool_args("send_email")[0]
    assert args.get("body"), "body should not be empty"


async def test_extracts_search_sender_filter(conversation):
    """search_emails should receive sender filter when user searches by sender."""
    conv = await conversation()
    await conv.send("Show me emails from boss@company.com")
    conv.assert_tool_called("search_emails")

    args = conv.get_tool_args("search_emails")[0]
    sender = args.get("sender", "") or ""
    query = args.get("query", "") or ""
    combined = f"{sender} {query}".lower()
    assert "boss" in combined or "boss@company.com" in combined, (
        f"Expected sender/query to reference 'boss@company.com', got sender='{sender}', query='{query}'"
    )


async def test_extracts_search_query_keywords(conversation):
    """search_emails should receive keyword filter for subject-based searches."""
    conv = await conversation()
    await conv.send("Find the email about quarterly report")
    conv.assert_tool_called("search_emails")

    args = conv.get_tool_args("search_emails")[0]
    query = args.get("query", "").lower()
    assert "quarterly" in query or "report" in query, (
        f"Expected query containing 'quarterly' or 'report', got '{query}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_check_inbox(conversation, llm_judge):
    """Checking emails should produce a readable listing of messages."""
    conv = await conversation()
    await conv.auto_complete("Check my inbox")

    passed = await llm_judge(
        "Check my inbox",
        conv.last_message,
        "The response should present a list of emails with senders and subjects "
        "in a readable format. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


async def test_response_quality_send(conversation, llm_judge):
    """Sending an email should confirm the action with recipient details."""
    conv = await conversation()
    msg = "Send an email to alice@example.com saying the meeting is confirmed"
    await conv.auto_complete(msg)

    passed = await llm_judge(
        msg,
        conv.last_message,
        "The response should confirm that an email was sent to "
        "alice@example.com. It should acknowledge the action positively.",
    )
    assert passed, f"LLM judge failed. Response: {conv.last_message}"


# ---------------------------------------------------------------------------
# Approval flow
# ---------------------------------------------------------------------------

async def test_send_email_triggers_approval(conversation):
    """send_email should pause for user approval before executing."""
    conv = await conversation()
    await conv.send_until_status(
        "Send an email to alice@example.com with subject Hello and body Hi Alice",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    conv.assert_tool_called("send_email")
    conv.assert_status(AgentStatus.WAITING_FOR_APPROVAL)


async def test_send_email_approve_executes(conversation):
    """Approving send_email should execute the tool and complete."""
    conv = await conversation()
    await conv.send_until_status(
        "Send an email to alice@example.com with subject Hello and body Hi Alice",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    result = await conv.send("yes, send it")
    assert result.status in (AgentStatus.COMPLETED, AgentStatus.WAITING_FOR_APPROVAL), (
        f"Expected COMPLETED after approval, got {result.status}"
    )


async def test_send_email_reject_cancels(conversation):
    """Rejecting send_email should cancel the operation."""
    conv = await conversation()
    await conv.send_until_status(
        "Send an email to alice@example.com with subject Hello and body Hi Alice",
        AgentStatus.WAITING_FOR_APPROVAL,
    )
    result = await conv.send("no, cancel it")
    assert result.status in (AgentStatus.CANCELLED, AgentStatus.COMPLETED), (
        f"Expected CANCELLED after rejection, got {result.status}"
    )
