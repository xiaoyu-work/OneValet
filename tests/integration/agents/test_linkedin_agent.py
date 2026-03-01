"""Integration tests for LinkedInComposioAgent.

Tests tool selection, argument extraction, and response quality for:
- create_post: Create a new post on LinkedIn
- get_my_profile: Get LinkedIn profile information
- connect_linkedin: Connect LinkedIn account via OAuth
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.communication]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Post on LinkedIn: Excited to share our latest milestone!", ["create_post"]),
    ("Create a LinkedIn post about my new job at Google", ["create_post"]),
    ("Show me my LinkedIn profile", ["get_my_profile"]),
    ("Connect my LinkedIn account", ["connect_linkedin"]),
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

async def test_extracts_post_text(orchestrator_factory):
    """create_post should receive the text content from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Post on LinkedIn: Thrilled to announce I just joined Anthropic as a Software Engineer!",
    )

    post_calls = [c for c in recorder.tool_calls if c["tool_name"] == "create_post"]
    assert post_calls, "create_post was never called"

    args = post_calls[0]["arguments"]
    text = args.get("text", "").lower()
    assert "anthropic" in text or "software engineer" in text, (
        f"Expected post text to reference Anthropic or role, got '{args.get('text')}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_profile(orchestrator_factory, llm_judge):
    """Getting profile info should produce a readable summary."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Show me my LinkedIn profile information"
    )

    passed = await llm_judge(
        "Show me my LinkedIn profile information",
        result.raw_message,
        "The response should present LinkedIn profile information such as "
        "name, headline, or other profile details. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result.raw_message}"
