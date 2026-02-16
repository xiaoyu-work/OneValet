"""
GitHubComposioAgent - Domain agent for GitHub operations via Composio.

Provides create/list issues, create/list pull requests, and search repositories
using the Composio OAuth proxy platform.
"""

import json
import os
import logging
from typing import Any, Dict

from onevalet import valet
from onevalet.agents.domain_agent import DomainAgent, DomainTool, DomainToolContext

from .client import ComposioClient

logger = logging.getLogger(__name__)

# Composio action ID constants for GitHub
_ACTION_CREATE_ISSUE = "GITHUB_CREATE_AN_ISSUE"
_ACTION_LIST_ISSUES = "GITHUB_LIST_REPOSITORY_ISSUES"
_ACTION_CREATE_PR = "GITHUB_CREATE_A_PULL_REQUEST"
_ACTION_LIST_PRS = "GITHUB_LIST_PULL_REQUESTS"
_ACTION_SEARCH_REPOS = "GITHUB_SEARCH_REPOSITORIES"
_APP_NAME = "github"


def _check_api_key() -> str | None:
    """Return error message if Composio API key is not configured, else None."""
    if not os.getenv("COMPOSIO_API_KEY"):
        return "Error: Composio API key not configured. Please add it in Settings."
    return None


# =============================================================================
# Tool executors
# =============================================================================

async def create_issue(args: dict, context: DomainToolContext) -> str:
    """Create a GitHub issue."""
    owner = args.get("owner", "")
    repo = args.get("repo", "")
    title = args.get("title", "")
    body = args.get("body", "")
    labels = args.get("labels", [])

    if not owner or not repo:
        return "Error: owner and repo are required."
    if not title:
        return "Error: title is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        params: Dict[str, Any] = {
            "owner": owner,
            "repo": repo,
            "title": title,
        }
        if body:
            params["body"] = body
        if labels:
            params["labels"] = labels

        data = await client.execute_action(_ACTION_CREATE_ISSUE, params=params)
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Issue created in {owner}/{repo}.\n\n{result}"
        return f"Failed to create issue: {result}"
    except Exception as e:
        logger.error(f"GitHub create_issue failed: {e}", exc_info=True)
        return f"Error creating GitHub issue: {e}"


async def list_issues(args: dict, context: DomainToolContext) -> str:
    """List issues in a GitHub repository."""
    owner = args.get("owner", "")
    repo = args.get("repo", "")
    state = args.get("state", "open")

    if not owner or not repo:
        return "Error: owner and repo are required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_ISSUES,
            params={"owner": owner, "repo": repo, "state": state},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Issues in {owner}/{repo} ({state}):\n\n{result}"
        return f"Failed to list issues: {result}"
    except Exception as e:
        logger.error(f"GitHub list_issues failed: {e}", exc_info=True)
        return f"Error listing GitHub issues: {e}"


async def create_pull_request(args: dict, context: DomainToolContext) -> str:
    """Create a GitHub pull request."""
    owner = args.get("owner", "")
    repo = args.get("repo", "")
    title = args.get("title", "")
    body = args.get("body", "")
    head = args.get("head", "")
    base = args.get("base", "")

    if not owner or not repo:
        return "Error: owner and repo are required."
    if not title:
        return "Error: title is required."
    if not head or not base:
        return "Error: head and base branches are required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        params: Dict[str, Any] = {
            "owner": owner,
            "repo": repo,
            "title": title,
            "head": head,
            "base": base,
        }
        if body:
            params["body"] = body

        data = await client.execute_action(_ACTION_CREATE_PR, params=params)
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Pull request created in {owner}/{repo}.\n\n{result}"
        return f"Failed to create pull request: {result}"
    except Exception as e:
        logger.error(f"GitHub create_pull_request failed: {e}", exc_info=True)
        return f"Error creating GitHub pull request: {e}"


async def list_pull_requests(args: dict, context: DomainToolContext) -> str:
    """List pull requests in a GitHub repository."""
    owner = args.get("owner", "")
    repo = args.get("repo", "")
    state = args.get("state", "open")

    if not owner or not repo:
        return "Error: owner and repo are required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_PRS,
            params={"owner": owner, "repo": repo, "state": state},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Pull requests in {owner}/{repo} ({state}):\n\n{result}"
        return f"Failed to list pull requests: {result}"
    except Exception as e:
        logger.error(f"GitHub list_pull_requests failed: {e}", exc_info=True)
        return f"Error listing GitHub pull requests: {e}"


async def search_repositories(args: dict, context: DomainToolContext) -> str:
    """Search GitHub repositories."""
    query = args.get("query", "")
    limit = args.get("limit", 10)

    if not query:
        return "Error: query is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SEARCH_REPOS,
            params={"q": query, "per_page": limit},
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Repositories matching '{query}':\n\n{result}"
        return f"Failed to search repositories: {result}"
    except Exception as e:
        logger.error(f"GitHub search_repositories failed: {e}", exc_info=True)
        return f"Error searching GitHub repositories: {e}"


async def connect_github(args: dict, context: DomainToolContext) -> str:
    """Initiate OAuth connection to GitHub via Composio."""
    entity_id = args.get("entity_id", "default")

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()

        # Check for existing active connection
        connections = await client.list_connections(entity_id=entity_id)
        connection_list = connections.get("items", connections.get("connections", []))
        for conn in connection_list:
            conn_app = (conn.get("appName") or conn.get("appUniqueId") or "").lower()
            conn_status = (conn.get("status") or "").upper()
            if conn_app == _APP_NAME and conn_status == "ACTIVE":
                return (
                    f"GitHub is already connected (account ID: {conn.get('id', 'unknown')}). "
                    f"You can use the other tools to interact with GitHub."
                )

        # Initiate new connection
        data = await client.initiate_connection(app_name=_APP_NAME, entity_id=entity_id)

        redirect = data.get("redirectUrl", data.get("redirect_url", ""))
        if redirect:
            return (
                f"To connect GitHub, please open this URL in your browser:\n\n"
                f"{redirect}\n\n"
                f"After completing the authorization, the connection will be active."
            )

        conn_id = data.get("id", data.get("connectedAccountId", ""))
        status = data.get("status", "")
        if status.upper() == "ACTIVE":
            return f"Successfully connected to GitHub. Connection ID: {conn_id}"
        return f"Connection initiated for GitHub. Status: {status}."
    except Exception as e:
        logger.error(f"GitHub connect failed: {e}", exc_info=True)
        return f"Error connecting to GitHub: {e}"


# =============================================================================
# Approval preview functions
# =============================================================================

async def _create_issue_preview(args: dict, context) -> str:
    owner = args.get("owner", "")
    repo = args.get("repo", "")
    title = args.get("title", "")
    body = args.get("body", "")
    preview = body[:100] + "..." if len(body) > 100 else body
    return f"Create GitHub issue?\n\nRepo: {owner}/{repo}\nTitle: {title}\nBody: {preview}"


async def _create_pr_preview(args: dict, context) -> str:
    owner = args.get("owner", "")
    repo = args.get("repo", "")
    title = args.get("title", "")
    head = args.get("head", "")
    base = args.get("base", "")
    return (
        f"Create GitHub pull request?\n\n"
        f"Repo: {owner}/{repo}\n"
        f"Title: {title}\n"
        f"Merge: {head} -> {base}"
    )


# =============================================================================
# Domain Agent
# =============================================================================

@valet(capabilities=["github", "code", "repository"])
class GitHubComposioAgent(DomainAgent):
    """Create and list issues, create and list pull requests, and search
    repositories on GitHub. Use when the user mentions GitHub, issues, PRs,
    pull requests, or repositories."""

    max_domain_turns = 5
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a GitHub assistant with access to GitHub tools via Composio.

Available tools:
- create_issue: Create a new issue in a GitHub repository.
- list_issues: List issues in a repository (open, closed, or all).
- create_pull_request: Create a new pull request.
- list_pull_requests: List pull requests in a repository.
- search_repositories: Search GitHub repositories by keyword.
- connect_github: Connect your GitHub account (OAuth).

Instructions:
1. If the user wants to create an issue, use create_issue with owner, repo, title, and body.
2. If the user wants to see issues, use list_issues with owner, repo, and optional state filter.
3. If the user wants to create a PR, use create_pull_request with owner, repo, title, head, and base.
4. If the user wants to see PRs, use list_pull_requests with owner, repo, and optional state filter.
5. If the user wants to find repositories, use search_repositories with a keyword query.
6. If GitHub is not yet connected, use connect_github first.
7. If the user's request is ambiguous or missing repository info, ask for clarification WITHOUT calling any tools.
8. After getting tool results, provide a clear summary to the user."""

    domain_tools = [
        DomainTool(
            name="create_issue",
            description="Create a new issue in a GitHub repository.",
            parameters={
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (user or organization)",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name",
                    },
                    "title": {
                        "type": "string",
                        "description": "Issue title",
                    },
                    "body": {
                        "type": "string",
                        "description": "Issue description (optional)",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels to add (optional)",
                    },
                },
                "required": ["owner", "repo", "title"],
            },
            executor=create_issue,
            needs_approval=True,
            risk_level="write",
            get_preview=_create_issue_preview,
        ),
        DomainTool(
            name="list_issues",
            description="List issues in a GitHub repository. Filter by state: open, closed, or all.",
            parameters={
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (user or organization)",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Issue state filter (default: open)",
                        "default": "open",
                    },
                },
                "required": ["owner", "repo"],
            },
            executor=list_issues,
        ),
        DomainTool(
            name="create_pull_request",
            description="Create a new pull request in a GitHub repository.",
            parameters={
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (user or organization)",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name",
                    },
                    "title": {
                        "type": "string",
                        "description": "Pull request title",
                    },
                    "body": {
                        "type": "string",
                        "description": "Pull request description (optional)",
                    },
                    "head": {
                        "type": "string",
                        "description": "Source branch name (the branch with changes)",
                    },
                    "base": {
                        "type": "string",
                        "description": "Target branch name (e.g. 'main')",
                    },
                },
                "required": ["owner", "repo", "title", "head", "base"],
            },
            executor=create_pull_request,
            needs_approval=True,
            risk_level="write",
            get_preview=_create_pr_preview,
        ),
        DomainTool(
            name="list_pull_requests",
            description="List pull requests in a GitHub repository. Filter by state: open, closed, or all.",
            parameters={
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (user or organization)",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "PR state filter (default: open)",
                        "default": "open",
                    },
                },
                "required": ["owner", "repo"],
            },
            executor=list_pull_requests,
        ),
        DomainTool(
            name="search_repositories",
            description="Search GitHub repositories by keyword.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (e.g. 'machine learning python')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            executor=search_repositories,
        ),
        DomainTool(
            name="connect_github",
            description="Connect your GitHub account via OAuth. Returns a URL to complete authorization.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity ID for multi-user setups (default: 'default')",
                        "default": "default",
                    },
                },
                "required": [],
            },
            executor=connect_github,
        ),
    ]
