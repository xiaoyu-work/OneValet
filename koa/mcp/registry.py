"""Official remote MCP server registry.

Declarative catalog of vendor-hosted MCP servers that replace hand-written API
agents.  Adding a server is data, not code: declare its endpoint, which
credential services can supply a token, and the pinned tool allowlist.

Two things the MCP protocol does *not* carry have to live here:

1. **Risk metadata.**  ``@tool(needs_approval=True, risk_level="write")`` is a
   Koa concept; a remote server only advertises name/description/schema.  Without
   the pins below, "create an issue" and "delete a repository" would look
   identical to the approval gate.
2. **A bounded tool surface.**  Vendor servers expose large, drifting catalogs
   (GitHub's ``all`` toolset is 100+ tools).  Pinning means vendor drift can only
   ever *shrink* capability, never silently grant new powers or blow up the
   prompt budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

#: Risk levels mirror ``koa.models.AgentTool.risk_level``.
RISK_READ = "read"
RISK_WRITE = "write"
RISK_DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class MCPToolSpec:
    """Risk classification for one pinned vendor tool."""

    name: str
    risk_level: str = RISK_READ
    needs_approval: bool = False
    #: Argument names redacted from logs and approval previews.
    sensitive_args: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.risk_level not in (RISK_READ, RISK_WRITE, RISK_DESTRUCTIVE):
            raise ValueError(f"{self.name}: invalid risk_level {self.risk_level!r}")
        # A write that nobody has to approve is almost always an oversight.
        if self.risk_level != RISK_READ and not self.needs_approval:
            raise ValueError(f"{self.name}: non-read tools must set needs_approval")


@dataclass(frozen=True)
class MCPServerSpec:
    """A vendor-hosted MCP server Koa can connect to on a tenant's behalf."""

    name: str
    url: str
    title: str
    #: Credential services checked in order for a usable token.  The first hit
    #: wins, so OAuth-connected accounts take precedence over pasted tokens.
    credential_services: Tuple[str, ...]
    tools: Tuple[MCPToolSpec, ...]
    #: Env var holding a shared fallback token for single-tenant/dev installs.
    fallback_env: Optional[str] = None
    #: Extra headers sent on every request (auth is added per tenant).
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def tool_specs(self) -> Dict[str, MCPToolSpec]:
        return {t.name: t for t in self.tools}

    @property
    def allowlist(self) -> Tuple[str, ...]:
        return tuple(t.name for t in self.tools)


# ---------------------------------------------------------------------------
# GitHub — https://api.githubcopilot.com/mcp/
# ---------------------------------------------------------------------------
# Replaces builtin_agents/composio/github_agent.py.  The pins below cover the
# same surface that agent exposed, so behaviour is preserved while the transport
# moves from the Composio proxy to GitHub's own server.
#
# Tool names verified against github/github-mcp-server pkg/github/__toolsnaps__,
# which is the authoritative per-tool snapshot directory.  `create_issue` is
# deliberately absent: it was superseded by the consolidated `issue_write`.

GITHUB = MCPServerSpec(
    name="github",
    url="https://api.githubcopilot.com/mcp/",
    title="GitHub",
    credential_services=("github",),
    fallback_env="GITHUB_TOKEN",
    tools=(
        MCPToolSpec("get_me"),
        MCPToolSpec("search_repositories"),
        MCPToolSpec("search_issues"),
        MCPToolSpec("search_pull_requests"),
        MCPToolSpec("search_code"),
        MCPToolSpec("get_file_contents"),
        MCPToolSpec("list_issues"),
        MCPToolSpec("issue_read"),
        MCPToolSpec("list_pull_requests"),
        MCPToolSpec("pull_request_read"),
        MCPToolSpec("list_commits"),
        MCPToolSpec("get_commit"),
        MCPToolSpec("list_branches"),
        MCPToolSpec("list_notifications"),
        # issue_write covers create/update — the old create_issue tool is gone.
        MCPToolSpec("issue_write", RISK_WRITE, needs_approval=True),
        MCPToolSpec("add_issue_comment", RISK_WRITE, needs_approval=True),
        MCPToolSpec("create_pull_request", RISK_WRITE, needs_approval=True),
        MCPToolSpec("pull_request_review_write", RISK_WRITE, needs_approval=True),
        MCPToolSpec("star_repository", RISK_WRITE, needs_approval=True),
        MCPToolSpec("merge_pull_request", RISK_DESTRUCTIVE, needs_approval=True),
    ),
)


# ---------------------------------------------------------------------------
# Notion — https://mcp.notion.com/mcp
# ---------------------------------------------------------------------------
# Replaces the hand-written NotionClient tool layer.  Notion's hosted server
# speaks OAuth 2.1 + PKCE, and also accepts an integration token as a bearer.
#
# Tool names verified against developers.notion.com/guides/mcp/mcp-supported-tools.
# Every tool carries the `notion-` prefix; only OpenAI clients see bare
# `search`/`fetch`, so the prefixed names are what we pin.  `notion-fetch` with
# the special id `self` covers workspace/user identity — there is no separate
# get-self tool.

NOTION = MCPServerSpec(
    name="notion",
    url="https://mcp.notion.com/mcp",
    title="Notion",
    credential_services=("notion",),
    fallback_env="NOTION_API_KEY",
    tools=(
        MCPToolSpec("notion-search"),
        MCPToolSpec("notion-fetch"),
        MCPToolSpec("notion-get-comments"),
        MCPToolSpec("notion-get-users"),
        MCPToolSpec("notion-get-teams"),
        MCPToolSpec("notion-query-data-sources"),
        MCPToolSpec("notion-query-database-view"),
        MCPToolSpec("notion-get-async-task"),
        MCPToolSpec("notion-create-pages", RISK_WRITE, needs_approval=True),
        MCPToolSpec("notion-update-page", RISK_WRITE, needs_approval=True),
        MCPToolSpec("notion-move-pages", RISK_WRITE, needs_approval=True),
        MCPToolSpec("notion-duplicate-page", RISK_WRITE, needs_approval=True),
        MCPToolSpec("notion-create-database", RISK_WRITE, needs_approval=True),
        MCPToolSpec("notion-update-data-source", RISK_WRITE, needs_approval=True),
        MCPToolSpec("notion-create-comment", RISK_WRITE, needs_approval=True),
    ),
)


SERVERS: Dict[str, MCPServerSpec] = {s.name: s for s in (GITHUB, NOTION)}


def get_server(name: str) -> Optional[MCPServerSpec]:
    """Look up a registered server spec by name."""
    return SERVERS.get(name)


def server_names() -> Tuple[str, ...]:
    return tuple(SERVERS)
