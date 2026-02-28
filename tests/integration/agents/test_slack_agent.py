"""Integration tests for SlackComposioAgent.

Tests tool selection, argument extraction, and response quality for:
- send_message: Send a message to a Slack channel or user
- fetch_messages: Fetch recent messages from a channel
- list_channels: List all available Slack channels
- find_users: Search for Slack users by name or email
- create_reminder: Create a Slack reminder
- connect_slack: Connect Slack account via OAuth
"""

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Send a Slack message to #engineering saying 'Deploy is done'", ["send_message"]),
    ("Show me the latest messages in the #general Slack channel", ["fetch_messages"]),
    ("List all Slack channels in the workspace", ["list_channels"]),
    ("Find the Slack user John Doe", ["find_users"]),
    ("Set a Slack reminder to check PR in 30 minutes", ["create_reminder"]),
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

async def test_extracts_message_fields(orchestrator_factory):
    """send_message should receive the channel and message text."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Send a Slack message to #engineering saying 'Build passed successfully'",
    )

    msg_calls = [c for c in recorder.tool_calls if c["tool_name"] == "send_message"]
    assert msg_calls, "send_message was never called"

    args = msg_calls[0]["arguments"]
    channel = args.get("channel", "").lower()
    assert "engineering" in channel, (
        f"Expected channel to contain 'engineering', got '{args.get('channel')}'"
    )
    text = args.get("text", "").lower()
    assert "build" in text or "passed" in text, (
        f"Expected text to contain message content, got '{args.get('text')}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_fetch(orchestrator_factory, llm_judge):
    """Fetching messages should produce a readable summary of the conversation."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Show me recent messages in #general on Slack"
    )

    passed = await llm_judge(
        "Show me recent messages in #general on Slack",
        result.raw_message,
        "The response should present Slack messages in a readable format, "
        "mentioning message content or senders. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result.raw_message}"
