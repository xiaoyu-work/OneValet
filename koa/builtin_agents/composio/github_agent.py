"""
GitHubComposioAgent - Agent for GitHub operations via Composio.

Provides create/list issues, create/list pull requests, and search repositories
using the Composio OAuth proxy platform.
"""

import logging
import os
from typing import Annotated, Any, Dict, List, Optional

from koa import valet
from koa.models import AgentToolContext
from koa.standard_agent import StandardAgent
from koa.tool_decorator import tool

from .client import ComposioClient

logger = logging.getLogger(__name__)

# Composio action ID constants for GitHub
_ACTION_CREATE_ISSUE = "GITHUB_CREATE_AN_ISSUE"
_ACTION_LIST_ISSUES = "GITHUB_LIST_REPOSITORY_ISSUES"
_ACTION_CREATE_PR = "GITHUB_CREATE_A_PULL_REQUEST"
_ACTION_LIST_PRS = "GITHUB_LIST_PULL_REQUESTS"
_ACTION_SEARCH_REPOS = "GITHUB_SEARCH_REPOSITORIES"
_ACTION_GET_REPO = "GITHUB_GET_A_REPOSITORY"
_ACTION_LIST_COMMITS = "GITHUB_LIST_COMMITS"
_ACTION_MERGE_PR = "GITHUB_MERGE_A_PULL_REQUEST"
_ACTION_LIST_BRANCHES = "GITHUB_LIST_BRANCHES"
_ACTION_STAR_REPO = "GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER"
_ACTION_LIST_NOTIFICATIONS = "GITHUB_LIST_NOTIFICATIONS_FOR_THE_AUTHENTICATED_USER"
_ACTION_CREATE_COMMENT = "GITHUB_CREATE_AN_ISSUE_COMMENT"
_ACTION_LIST_USER_REPOS = "GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER"
_APP_NAME = "github"


def _check_api_key() -> Optional[str]:
    """Return error message if Composio API key is not configured, else None."""
    if not os.getenv("COMPOSIO_API_KEY"):
        return "Error: Composio API key not configured. Please add it in Settings."
    return None


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
# Tool executors
# =============================================================================


@tool(needs_approval=True, risk_level="write", get_preview=_create_issue_preview)
async def create_issue(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    title: Annotated[str, "Issue title"],
    body: Annotated[str, "Issue description"] = "",
    labels: Annotated[List[str], "Labels to add"] = [],
    *,
    context: AgentToolContext,
) -> str:
    """Create a new issue in a GitHub repository."""

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

        data = await client.execute_action(
            _ACTION_CREATE_ISSUE, params=params, entity_id=context.tenant_id or "default"
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Issue created in {owner}/{repo}.\n\n{result}"
        return f"Failed to create issue: {result}"
    except Exception as e:
        logger.error(f"GitHub create_issue failed: {e}", exc_info=True)
        return f"Error creating GitHub issue: {e}"


@tool
async def list_issues(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    state: Annotated[str, "Issue state filter: open, closed, or all"] = "open",
    *,
    context: AgentToolContext,
) -> str:
    """List issues in a GitHub repository. Filter by state: open, closed, or all."""

    if not owner or not repo:
        return "Error: owner and repo are required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_ISSUES,
            params={"owner": owner, "repo": repo, "state": state},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Issues in {owner}/{repo} ({state}):\n\n{result}"
        return f"Failed to list issues: {result}"
    except Exception as e:
        logger.error(f"GitHub list_issues failed: {e}", exc_info=True)
        return f"Error listing GitHub issues: {e}"


@tool(needs_approval=True, risk_level="write", get_preview=_create_pr_preview)
async def create_pull_request(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    title: Annotated[str, "Pull request title"],
    head: Annotated[str, "Source branch name (the branch with changes)"],
    base: Annotated[str, "Target branch name (e.g. 'main')"],
    body: Annotated[str, "Pull request description"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Create a new pull request in a GitHub repository."""

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

        data = await client.execute_action(
            _ACTION_CREATE_PR, params=params, entity_id=context.tenant_id or "default"
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Pull request created in {owner}/{repo}.\n\n{result}"
        return f"Failed to create pull request: {result}"
    except Exception as e:
        logger.error(f"GitHub create_pull_request failed: {e}", exc_info=True)
        return f"Error creating GitHub pull request: {e}"


@tool
async def list_pull_requests(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    state: Annotated[str, "PR state filter: open, closed, or all"] = "open",
    *,
    context: AgentToolContext,
) -> str:
    """List pull requests in a GitHub repository. Filter by state: open, closed, or all."""

    if not owner or not repo:
        return "Error: owner and repo are required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_PRS,
            params={"owner": owner, "repo": repo, "state": state},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Pull requests in {owner}/{repo} ({state}):\n\n{result}"
        return f"Failed to list pull requests: {result}"
    except Exception as e:
        logger.error(f"GitHub list_pull_requests failed: {e}", exc_info=True)
        return f"Error listing GitHub pull requests: {e}"


@tool
async def search_repositories(
    query: Annotated[str, "Search keywords (e.g. 'machine learning python')"],
    limit: Annotated[int, "Max results to return"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """Search GitHub repositories by keyword."""

    if not query:
        return "Error: query is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_SEARCH_REPOS,
            params={"q": query, "per_page": limit},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Repositories matching '{query}':\n\n{result}"
        return f"Failed to search repositories: {result}"
    except Exception as e:
        logger.error(f"GitHub search_repositories failed: {e}", exc_info=True)
        return f"Error searching GitHub repositories: {e}"


@tool
async def get_repository(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    *,
    context: AgentToolContext,
) -> str:
    """Get details about a GitHub repository."""

    if not owner or not repo:
        return "Error: owner and repo are required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_GET_REPO,
            params={"owner": owner, "repo": repo},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Repository {owner}/{repo}:\n\n{result}"
        return f"Failed to get repository: {result}"
    except Exception as e:
        logger.error(f"GitHub get_repository failed: {e}", exc_info=True)
        return f"Error getting GitHub repository: {e}"


@tool
async def list_commits(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    sha: Annotated[str, "Branch name or commit SHA to list commits from"] = "",
    per_page: Annotated[int, "Number of commits to return"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """List recent commits in a GitHub repository."""

    if not owner or not repo:
        return "Error: owner and repo are required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        params: Dict[str, Any] = {
            "owner": owner,
            "repo": repo,
            "per_page": per_page,
        }
        if sha:
            params["sha"] = sha

        data = await client.execute_action(
            _ACTION_LIST_COMMITS,
            params=params,
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Commits in {owner}/{repo}:\n\n{result}"
        return f"Failed to list commits: {result}"
    except Exception as e:
        logger.error(f"GitHub list_commits failed: {e}", exc_info=True)
        return f"Error listing GitHub commits: {e}"


@tool(needs_approval=True, risk_level="write")
async def merge_pull_request(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    pull_number: Annotated[int, "Pull request number to merge"],
    merge_method: Annotated[str, "Merge method: 'merge', 'squash', or 'rebase'"] = "merge",
    *,
    context: AgentToolContext,
) -> str:
    """Merge a pull request in a GitHub repository."""

    if not owner or not repo:
        return "Error: owner and repo are required."
    if not pull_number:
        return "Error: pull_number is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_MERGE_PR,
            params={
                "owner": owner,
                "repo": repo,
                "pull_number": pull_number,
                "merge_method": merge_method,
            },
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Pull request #{pull_number} merged in {owner}/{repo}.\n\n{result}"
        return f"Failed to merge pull request: {result}"
    except Exception as e:
        logger.error(f"GitHub merge_pull_request failed: {e}", exc_info=True)
        return f"Error merging GitHub pull request: {e}"


@tool
async def list_branches(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    *,
    context: AgentToolContext,
) -> str:
    """List branches in a GitHub repository."""

    if not owner or not repo:
        return "Error: owner and repo are required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_BRANCHES,
            params={"owner": owner, "repo": repo},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Branches in {owner}/{repo}:\n\n{result}"
        return f"Failed to list branches: {result}"
    except Exception as e:
        logger.error(f"GitHub list_branches failed: {e}", exc_info=True)
        return f"Error listing GitHub branches: {e}"


@tool(needs_approval=True, risk_level="write")
async def star_repo(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    *,
    context: AgentToolContext,
) -> str:
    """Star a GitHub repository for the authenticated user."""

    if not owner or not repo:
        return "Error: owner and repo are required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_STAR_REPO,
            params={"owner": owner, "repo": repo},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Starred {owner}/{repo}.\n\n{result}"
        return f"Failed to star repository: {result}"
    except Exception as e:
        logger.error(f"GitHub star_repo failed: {e}", exc_info=True)
        return f"Error starring GitHub repository: {e}"


@tool
async def list_notifications(
    all: Annotated[bool, "If True, show all notifications including read ones"] = False,
    *,
    context: AgentToolContext,
) -> str:
    """List GitHub notifications for the authenticated user."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_NOTIFICATIONS,
            params={"all": all},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"GitHub notifications:\n\n{result}"
        return f"Failed to list notifications: {result}"
    except Exception as e:
        logger.error(f"GitHub list_notifications failed: {e}", exc_info=True)
        return f"Error listing GitHub notifications: {e}"


@tool(needs_approval=True, risk_level="write")
async def create_issue_comment(
    owner: Annotated[str, "Repository owner (user or organization)"],
    repo: Annotated[str, "Repository name"],
    issue_number: Annotated[int, "Issue or pull request number to comment on"],
    body: Annotated[str, "Comment text"],
    *,
    context: AgentToolContext,
) -> str:
    """Create a comment on a GitHub issue or pull request."""

    if not owner or not repo:
        return "Error: owner and repo are required."
    if not issue_number:
        return "Error: issue_number is required."
    if not body:
        return "Error: body is required."
    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_CREATE_COMMENT,
            params={
                "owner": owner,
                "repo": repo,
                "issue_number": issue_number,
                "body": body,
            },
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Comment added to {owner}/{repo}#{issue_number}.\n\n{result}"
        return f"Failed to create comment: {result}"
    except Exception as e:
        logger.error(f"GitHub create_issue_comment failed: {e}", exc_info=True)
        return f"Error creating GitHub issue comment: {e}"


@tool
async def list_my_repos(
    per_page: Annotated[int, "Number of repositories to return"] = 10,
    sort: Annotated[str, "Sort by: 'created', 'updated', 'pushed', or 'full_name'"] = "updated",
    *,
    context: AgentToolContext,
) -> str:
    """List repositories for the authenticated GitHub user."""

    if err := _check_api_key():
        return err

    try:
        client = ComposioClient()
        data = await client.execute_action(
            _ACTION_LIST_USER_REPOS,
            params={"per_page": per_page, "sort": sort},
            entity_id=context.tenant_id or "default",
        )
        result = ComposioClient.format_action_result(data)
        if data.get("successfull") or data.get("successful"):
            return f"Your GitHub repositories:\n\n{result}"
        return f"Failed to list repositories: {result}"
    except Exception as e:
        logger.error(f"GitHub list_my_repos failed: {e}", exc_info=True)
        return f"Error listing GitHub repositories: {e}"


@tool
async def connect_github(
    entity_id: Annotated[str, "Entity ID for multi-user setups"] = "default",
    *,
    context: AgentToolContext,
) -> str:
    """Connect your GitHub account via OAuth. Returns a URL to complete authorization."""

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
# Agent
# =============================================================================


@valet(domain="productivity")
class GitHubComposioAgent(StandardAgent):
    """Create and list issues, create and list pull requests, and search
    repositories on GitHub. Use when the user mentions GitHub, issues, PRs,
    pull requests, or repositories."""

    max_turns = 15
    tool_timeout = 60.0

    domain_system_prompt = """\
You are a GitHub assistant with access to GitHub tools via Composio.

Available tools:
- create_issue: Create a new issue in a GitHub repository.
- list_issues: List issues in a repository (open, closed, or all).
- create_pull_request: Create a new pull request.
- list_pull_requests: List pull requests in a repository.
- search_repositories: Search GitHub repositories by keyword.
- get_repository: Get details about a specific repository.
- list_commits: List recent commits in a repository.
- merge_pull_request: Merge a pull request (merge, squash, or rebase).
- list_branches: List branches in a repository.
- star_repo: Star a repository for the authenticated user.
- list_notifications: List GitHub notifications.
- create_issue_comment: Comment on an issue or pull request.
- list_my_repos: List the authenticated user's repositories.
- connect_github: Connect your GitHub account (OAuth).

Instructions:
1. If the user wants to create an issue, use create_issue with owner, repo, title, and body.
2. If the user wants to see issues, use list_issues with owner, repo, and optional state filter.
3. If the user wants to create a PR, use create_pull_request with owner, repo, title, head, and base.
4. If the user wants to see PRs, use list_pull_requests with owner, repo, and optional state filter.
5. If the user wants to find repositories, use search_repositories with a keyword query.
6. If the user wants details about a repo, use get_repository with owner and repo.
7. If the user wants to see recent commits, use list_commits with owner and repo.
8. If the user wants to merge a PR, use merge_pull_request with owner, repo, pull_number, and optional merge_method.
9. If the user wants to see branches, use list_branches with owner and repo.
10. If the user wants to star a repo, use star_repo with owner and repo.
11. If the user wants to check notifications, use list_notifications.
12. If the user wants to comment on an issue/PR, use create_issue_comment with owner, repo, issue_number, and body.
13. If the user wants to see their own repos, use list_my_repos.
14. If GitHub is not yet connected, use connect_github first.
15. If the user's request is ambiguous or missing repository info, ask for clarification WITHOUT calling any tools.
16. After getting tool results, provide a clear summary to the user."""

    tools = (
        create_issue,
        list_issues,
        create_pull_request,
        list_pull_requests,
        search_repositories,
        get_repository,
        list_commits,
        merge_pull_request,
        list_branches,
        star_repo,
        list_notifications,
        create_issue_comment,
        list_my_repos,
        connect_github,
    )
