"""Integration tests for NotionAgent — tool selection, argument extraction, response quality.

NotionAgent tools:
  notion_search, notion_read_page, notion_query_database,
  notion_create_page, notion_update_page
"""

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Search for meeting notes in Notion", ["notion_search"]),
    ("Find my project tracker in Notion", ["notion_search"]),
    ("Read the Meeting Notes page in Notion", ["notion_search", "notion_read_page"]),
    ("Create a new Notion page called Weekly Review", ["notion_create_page"]),
    ("Add notes to the Meeting Notes page in Notion", ["notion_update_page"]),
    ("Show me what's in my Notion workspace", ["notion_search"]),
    ("Query the tasks database in Notion", ["notion_search", "notion_query_database"]),
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

async def test_search_extracts_query(orchestrator_factory):
    """notion_search should receive a short keyword from the user's request."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Search for meeting notes in Notion")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "notion_search"]
    assert calls, "Expected notion_search to be called"

    args = calls[0]["arguments"]
    query = args.get("query", "")
    assert query, f"Expected a search query, got {args}"
    # The keyword should relate to "meeting notes"
    assert any(w in query.lower() for w in ["meeting", "notes"]), (
        f"Expected query related to 'meeting notes', got '{query}'"
    )


async def test_create_page_extracts_title(orchestrator_factory):
    """notion_create_page should receive the page title."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Create a new Notion page called Weekly Review with summary of this week",
    )

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "notion_create_page"]
    assert calls, "Expected notion_create_page to be called"

    args = calls[0]["arguments"]
    title = args.get("title", "")
    assert title, f"Expected a title, got {args}"
    assert "weekly" in title.lower() or "review" in title.lower(), (
        f"Expected title containing 'Weekly Review', got '{title}'"
    )


async def test_update_page_extracts_title_and_content(orchestrator_factory):
    """notion_update_page should receive the page title and content to append."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Add 'Discussed Q1 roadmap' to my Meeting Notes page in Notion",
    )

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "notion_update_page"]
    assert calls, "Expected notion_update_page to be called"

    args = calls[0]["arguments"]
    page_title = args.get("page_title", "")
    content = args.get("content", "")
    assert page_title, f"Expected page_title, got {args}"
    assert content, f"Expected content, got {args}"


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_search(orchestrator_factory, llm_judge):
    """Searching Notion should present results clearly."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Search for meeting notes in Notion"
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Search for meeting notes in Notion",
        response=response,
        criteria=(
            "The response should present Notion search results with page titles. "
            "It should mention 'Meeting Notes' since that is in the canned data. "
            "It should be well-formatted."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"


async def test_response_quality_create_page(orchestrator_factory, llm_judge):
    """Creating a page should confirm what was created."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user",
        "Create a new Notion page called Project Ideas",
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Create a new Notion page called Project Ideas",
        response=response,
        criteria=(
            "The response should confirm that a Notion page was created. "
            "It may include a URL or page ID. It should not ask unnecessary "
            "follow-up questions."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"
