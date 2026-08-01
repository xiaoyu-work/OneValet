"""GitHubMCPAgent — GitHub via GitHub's own hosted MCP server.

Replaces the Composio-proxied GitHubComposioAgent: the tools now come from
https://api.githubcopilot.com/mcp/ directly, so tokens and request payloads no
longer transit a third party, and GitHub maintains the tool surface.
"""

from koa import valet

from ..mcp_backed import MCPBackedAgent


@valet(domain="productivity", requires_service=["github"])
class GitHubMCPAgent(MCPBackedAgent):
    """Work with GitHub repositories, issues, pull requests, commits, and code search. Use when the user mentions GitHub, repos, issues, PRs, or code review."""

    mcp_server = "github"
    max_turns = 15

    domain_system_prompt = """\
You are a GitHub assistant backed by GitHub's official MCP server.

Guidance:
1. Most tools need `owner` and `repo`. If the user names a repo as "owner/repo",
   split it. If the repo is ambiguous, ask before calling a tool.
2. To find repositories, use search_repositories. To find work items, prefer
   search_issues / search_pull_requests over listing everything.
3. Read a single item with issue_read or pull_request_read; these take a
   `method` argument that selects what to fetch (details, comments, diff, ...).
4. issue_write creates and updates issues. Creating or merging anything requires
   user approval — state clearly what you are about to do.
5. After tool results, summarise plainly. Refer to items by number and title
   rather than dumping raw JSON."""
