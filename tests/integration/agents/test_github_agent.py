"""Integration tests for GitHubComposioAgent.

Tests tool selection, argument extraction, and response quality for:
- create_issue: Create a new issue in a GitHub repository
- list_issues: List issues in a repository (open, closed, all)
- create_pull_request: Create a new pull request
- list_pull_requests: List pull requests in a repository
- search_repositories: Search GitHub repositories by keyword
- connect_github: Connect GitHub account via OAuth
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.productivity]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Create an issue in org/repo titled 'Login bug' about the auth failure", ["create_issue"]),
    ("Show me the open issues in facebook/react", ["list_issues"]),
    ("Create a PR in org/repo to merge feature-branch into main titled 'Fix login'", ["create_pull_request"]),
    ("List open pull requests in vercel/next.js", ["list_pull_requests"]),
    ("Search GitHub for machine learning Python repos", ["search_repositories"]),
    ("Connect my GitHub account", ["connect_github"]),
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

async def test_extracts_issue_fields(orchestrator_factory):
    """create_issue should receive owner, repo, and title from the user message."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Create an issue in facebook/react titled 'Hydration mismatch on SSR'",
    )

    issue_calls = [c for c in recorder.tool_calls if c["tool_name"] == "create_issue"]
    assert issue_calls, "create_issue was never called"

    args = issue_calls[0]["arguments"]
    assert args.get("owner", "").lower() == "facebook", (
        f"Expected owner='facebook', got {args.get('owner')}"
    )
    assert args.get("repo", "").lower() == "react", (
        f"Expected repo='react', got {args.get('repo')}"
    )
    assert args.get("title"), "title should not be empty"


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_list_issues(orchestrator_factory, llm_judge):
    """Listing issues should produce a readable summary of the issues."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Show open issues in org/repo")

    passed = await llm_judge(
        "Show open issues in org/repo",
        result.raw_message,
        "The response should present a list or summary of GitHub issues. "
        "It should mention issue titles or numbers and not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result.raw_message}"
