"""Integration tests for DiscordComposioAgent.

Tests tool selection, argument extraction, and response quality for:
- send_message: Send a message to a Discord channel
- list_channels: List all channels in a Discord guild/server
- list_servers: List all Discord guilds/servers the user belongs to
- connect_discord: Connect Discord account via OAuth
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.communication]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Send a message on Discord to channel 123456 saying 'Hello everyone'", ["send_message"]),
    ("Show me all channels in my Discord server 789012", ["list_channels"]),
    ("List all my Discord servers", ["list_servers"]),
    ("Connect my Discord account via OAuth", ["connect_discord"]),
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
    """send_message should receive channel_id and content from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Send a Discord message to channel 987654321 saying 'Meeting starts now'",
    )

    msg_calls = [c for c in recorder.tool_calls if c["tool_name"] == "send_message"]
    assert msg_calls, "send_message was never called"

    args = msg_calls[0]["arguments"]
    assert "987654321" in str(args.get("channel_id", "")), (
        f"Expected channel_id to contain '987654321', got {args.get('channel_id')}"
    )
    content = args.get("content", "").lower()
    assert "meeting" in content, (
        f"Expected content to contain 'meeting', got '{args.get('content')}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_list_servers(orchestrator_factory, llm_judge):
    """Listing servers should produce a readable summary of Discord servers."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Show me all my Discord servers"
    )

    passed = await llm_judge(
        "Show me all my Discord servers",
        result.raw_message,
        "The response should present a list or summary of Discord servers/guilds. "
        "It should mention server names and not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result.raw_message}"
