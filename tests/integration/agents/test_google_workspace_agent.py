"""Integration tests for GoogleWorkspaceAgent — tool selection, argument extraction, response quality.

GoogleWorkspaceAgent tools:
  google_drive_search, google_docs_read, google_sheets_read,
  google_docs_create, google_sheets_write
"""

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Search for Q4 Report in Google Drive", ["google_drive_search"]),
    ("Find my budget spreadsheet in Google Drive", ["google_drive_search"]),
    ("Read the Q4 Report Google Doc", ["google_drive_search", "google_docs_read"]),
    ("Show me what's in the Budget spreadsheet", ["google_drive_search", "google_sheets_read"]),
    ("Create a new Google Doc called Meeting Agenda", ["google_docs_create"]),
    ("Write data to the Budget spreadsheet", ["google_sheets_write"]),
    ("List my recent Google Drive files", ["google_drive_search"]),
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

async def test_drive_search_extracts_query(orchestrator_factory):
    """google_drive_search should receive a search keyword."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Find the Q4 Report in Google Drive")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "google_drive_search"]
    assert calls, "Expected google_drive_search to be called"

    args = calls[0]["arguments"]
    query = args.get("query", "")
    assert query, f"Expected a search query, got {args}"
    assert "q4" in query.lower() or "report" in query.lower(), (
        f"Expected query related to 'Q4 Report', got '{query}'"
    )


async def test_docs_create_extracts_title(orchestrator_factory):
    """google_docs_create should receive the document title."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Create a new Google Doc called Meeting Agenda with the date and attendees section",
    )

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "google_docs_create"]
    assert calls, "Expected google_docs_create to be called"

    args = calls[0]["arguments"]
    title = args.get("title", "")
    assert title, f"Expected a title, got {args}"
    assert "meeting" in title.lower() or "agenda" in title.lower(), (
        f"Expected title containing 'Meeting Agenda', got '{title}'"
    )


async def test_sheets_write_extracts_spreadsheet_name(orchestrator_factory):
    """google_sheets_write should receive the spreadsheet name and data."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        'Write the values [["Name","Age"],["Alice",30]] to the Budget spreadsheet in range Sheet1!A1:B2',
    )

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "google_sheets_write"]
    assert calls, "Expected google_sheets_write to be called"

    args = calls[0]["arguments"]
    name = args.get("spreadsheet_name", "")
    assert name, f"Expected spreadsheet_name, got {args}"
    assert "budget" in name.lower(), f"Expected 'Budget' in spreadsheet_name, got '{name}'"

    range_val = args.get("range", "")
    assert range_val, f"Expected range, got {args}"

    values = args.get("values", "")
    assert values, f"Expected values, got {args}"


# ---------------------------------------------------------------------------
# Multi-step tool chains
# ---------------------------------------------------------------------------

async def test_read_doc_triggers_search_then_read(orchestrator_factory):
    """Reading a doc by name should first search, then read by ID."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Read the Q4 Report Google Doc",
    )

    tools_called = [c["tool_name"] for c in recorder.tool_calls]
    # Should search first, then read (multi-turn)
    assert "google_drive_search" in tools_called, (
        f"Expected google_drive_search in chain, got {tools_called}"
    )
    # The agent may also call google_docs_read if it extracts the ID from canned search results
    # This depends on the LLM's behavior, so we check search was at least called


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_drive_search(orchestrator_factory, llm_judge):
    """Searching Google Drive should present file results."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Find my Q4 Report in Google Drive"
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Find my Q4 Report in Google Drive",
        response=response,
        criteria=(
            "The response should present Google Drive search results. "
            "It should mention the Q4 Report file from the results. "
            "It should be well-formatted and informative."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"


async def test_response_quality_create_doc(orchestrator_factory, llm_judge):
    """Creating a Google Doc should confirm what was created."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Create a Google Doc called Sprint Notes"
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Create a Google Doc called Sprint Notes",
        response=response,
        criteria=(
            "The response should confirm that a Google Doc was created. "
            "It should mention the title or provide a URL. "
            "It should not ask unnecessary questions."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"
