"""Tests for the official remote MCP server integration (GitHub, Notion)."""

import pytest

from koa.mcp.client import MockMCPClient
from koa.mcp.models import MCPTool
from koa.mcp.provider import MCPManager, MCPToolProvider
from koa.mcp.registry import (
    GITHUB,
    NOTION,
    SERVERS,
    MCPToolSpec,
    get_server,
)
from koa.mcp.tenant_tools import (
    MCPCredentialError,
    build_client,
    extract_token,
    load_tools,
    resolve_token,
)


def _tool(name: str) -> MCPTool:
    return MCPTool(
        name=name,
        description=f"{name} description",
        input_schema={"type": "object", "properties": {}},
        server_name="test-server",
    )


class FakeCredentialStore:
    """Minimal CredentialStore stand-in keyed by (tenant, service, account)."""

    def __init__(self, data=None):
        self._data = data or {}

    async def get(self, tenant_id, service, account_name="primary"):
        return self._data.get((tenant_id, service, account_name))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_servers(self):
        assert set(SERVERS) == {"github", "notion"}
        assert get_server("github") is GITHUB
        assert get_server("nope") is None

    def test_endpoints(self):
        assert GITHUB.url == "https://api.githubcopilot.com/mcp/"
        assert NOTION.url == "https://mcp.notion.com/mcp"

    def test_every_write_tool_requires_approval(self):
        for spec in SERVERS.values():
            for tool in spec.tools:
                if tool.risk_level != "read":
                    assert tool.needs_approval, f"{tool.name} is a write without approval"

    def test_write_without_approval_is_rejected(self):
        with pytest.raises(ValueError, match="needs_approval"):
            MCPToolSpec("danger", "write", needs_approval=False)

    def test_invalid_risk_level_is_rejected(self):
        with pytest.raises(ValueError, match="risk_level"):
            MCPToolSpec("weird", "catastrophic", needs_approval=True)

    def test_destructive_tools_are_marked(self):
        assert GITHUB.tool_specs["merge_pull_request"].risk_level == "destructive"

    def test_notion_tools_keep_vendor_prefix(self):
        # Only OpenAI clients see bare search/fetch; we must pin the real names.
        assert "notion-search" in NOTION.tool_specs
        assert "search" not in NOTION.tool_specs

    def test_allowlist_matches_specs(self):
        assert set(GITHUB.allowlist) == set(GITHUB.tool_specs)


# ---------------------------------------------------------------------------
# Provider: allowlist + risk metadata
# ---------------------------------------------------------------------------


class TestProviderPinning:
    async def test_unpinned_tools_are_dropped(self):
        client = MockMCPClient(
            name="github",
            tools=[_tool("search_repositories"), _tool("delete_everything")],
        )
        await client.connect()
        provider = MCPToolProvider(client, tool_specs=GITHUB.tool_specs)
        tools = await provider.discover_tools()

        names = {t.name for t in tools}
        assert names == {"mcp__github__search_repositories"}

    async def test_risk_metadata_is_applied(self):
        client = MockMCPClient(
            name="github",
            tools=[
                _tool("search_repositories"),
                _tool("issue_write"),
                _tool("merge_pull_request"),
            ],
        )
        await client.connect()
        provider = MCPToolProvider(client, tool_specs=GITHUB.tool_specs)
        by_name = {t.name: t for t in await provider.discover_tools()}

        read = by_name["mcp__github__search_repositories"]
        assert read.risk_level == "read"
        assert read.needs_approval is False
        assert read.read_only is True

        write = by_name["mcp__github__issue_write"]
        assert write.risk_level == "write"
        assert write.needs_approval is True
        assert write.mutates_user_data is True

        destructive = by_name["mcp__github__merge_pull_request"]
        assert destructive.risk_level == "destructive"
        assert destructive.needs_approval is True

    async def test_without_specs_all_tools_pass_through(self):
        client = MockMCPClient(name="custom", tools=[_tool("anything")])
        await client.connect()
        provider = MCPToolProvider(client)
        tools = await provider.discover_tools()
        assert [t.name for t in tools] == ["mcp__custom__anything"]

    async def test_missing_pinned_tool_is_logged(self, caplog):
        client = MockMCPClient(name="github", tools=[_tool("search_repositories")])
        await client.connect()
        provider = MCPToolProvider(client, tool_specs=GITHUB.tool_specs)
        with caplog.at_level("WARNING"):
            await provider.discover_tools()
        assert "Pinned tool(s) missing" in caplog.text


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    async def test_same_server_two_tenants_do_not_collide(self):
        client_a = MockMCPClient(name="github", tools=[_tool("search_repositories")])
        client_b = MockMCPClient(name="github", tools=[_tool("list_issues")])

        manager = MCPManager()
        await manager.add_server(client_a, tenant_id="tenant-a")
        await manager.add_server(client_b, tenant_id="tenant-b")

        # Without tenant keying the second add would have evicted the first.
        assert manager.get_provider("github", "tenant-a") is not None
        assert manager.get_provider("github", "tenant-b") is not None

        assert manager.get_all_tool_names("tenant-a") == ["mcp__github__search_repositories"]
        assert manager.get_all_tool_names("tenant-b") == ["mcp__github__list_issues"]

    async def test_removing_one_tenant_leaves_the_other(self):
        manager = MCPManager()
        await manager.add_server(
            MockMCPClient(name="github", tools=[_tool("list_issues")]), tenant_id="a"
        )
        await manager.add_server(
            MockMCPClient(name="github", tools=[_tool("list_issues")]), tenant_id="b"
        )

        await manager.remove_server("github", tenant_id="a")

        assert manager.get_provider("github", "a") is None
        assert manager.get_provider("github", "b") is not None

    async def test_get_all_tools_without_tenant_spans_tenants(self):
        manager = MCPManager()
        await manager.add_server(
            MockMCPClient(name="github", tools=[_tool("list_issues")]), tenant_id="a"
        )
        await manager.add_server(
            MockMCPClient(name="notion", tools=[_tool("notion-search")]), tenant_id="b"
        )
        assert len(manager.get_all_tools()) == 2
        assert len(manager.get_all_tools("a")) == 1


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_extract_token_variants(self):
        assert extract_token({"access_token": "a"}) == "a"
        assert extract_token({"api_key": "k"}) == "k"
        # CredentialStore.get() may wrap the payload in an envelope.
        assert extract_token({"credentials": {"access_token": "inner"}}) == "inner"
        assert extract_token({}) is None
        assert extract_token(None) is None

    async def test_tenant_credential_is_used(self):
        store = FakeCredentialStore({("t1", "github", "primary"): {"access_token": "tok"}})
        assert await resolve_token(GITHUB, "t1", store) == "tok"

    async def test_tenant_credential_beats_env_fallback(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "env-token")
        store = FakeCredentialStore({("t1", "github", "primary"): {"access_token": "tenant"}})
        assert await resolve_token(GITHUB, "t1", store) == "tenant"

    async def test_env_fallback_when_tenant_has_none(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "env-token")
        assert await resolve_token(GITHUB, "t1", FakeCredentialStore()) == "env-token"

    async def test_no_credential_returns_none(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        assert await resolve_token(GITHUB, "t1", FakeCredentialStore()) is None

    async def test_broken_store_does_not_raise(self, monkeypatch):
        monkeypatch.delenv("NOTION_API_KEY", raising=False)

        class Broken:
            async def get(self, *a, **k):
                raise RuntimeError("db down")

        assert await resolve_token(NOTION, "t1", Broken()) is None

    def test_build_client_sets_bearer_header(self):
        client = build_client(GITHUB, "secret-token")
        assert client.config.headers["Authorization"] == "Bearer secret-token"
        assert client.config.url == GITHUB.url
        assert client.config.transport.value == "streamable_http"

    async def test_load_tools_without_credential_raises(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with pytest.raises(MCPCredentialError, match="not connected"):
            await load_tools("github", "t1", FakeCredentialStore())

    async def test_load_tools_unknown_server_raises(self):
        with pytest.raises(ValueError, match="unknown MCP server"):
            await load_tools("nope", "t1", FakeCredentialStore())


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class TestMCPBackedAgents:
    def test_agents_are_registered(self):
        from koa.agents.decorator import AGENT_REGISTRY
        from koa.builtin_agents.github import GitHubMCPAgent  # noqa: F401
        from koa.builtin_agents.notion import NotionMCPAgent  # noqa: F401

        assert "GitHubMCPAgent" in AGENT_REGISTRY
        assert "NotionMCPAgent" in AGENT_REGISTRY

    def test_agents_point_at_registered_servers(self):
        from koa.builtin_agents.github import GitHubMCPAgent
        from koa.builtin_agents.notion import NotionMCPAgent

        assert get_server(GitHubMCPAgent.mcp_server) is GITHUB
        assert get_server(NotionMCPAgent.mcp_server) is NOTION

    def test_tools_are_per_instance_not_class_level(self):
        from koa.builtin_agents.github import GitHubMCPAgent

        # Class-level tools must stay empty: MCP tools are per-tenant, so
        # sharing them on the class would leak one tenant's session to another.
        assert GitHubMCPAgent.tools == ()
        agent = GitHubMCPAgent(tenant_id="t1")
        agent.tools = ("sentinel",)
        assert GitHubMCPAgent(tenant_id="t2").tools == ()

    async def test_missing_credential_yields_friendly_message(self, monkeypatch):
        from koa.builtin_agents.notion import NotionMCPAgent
        from koa.message import Message

        monkeypatch.delenv("NOTION_API_KEY", raising=False)
        agent = NotionMCPAgent(tenant_id="t1")
        agent.context_hints = {"credential_store": FakeCredentialStore()}

        result = await agent.reply(Message(name="user", content="find my notes", role="user"))
        assert "Notion is not connected" in result.raw_message

    def test_requires_service_gates_the_agent(self):
        from koa.agents.decorator import AGENT_REGISTRY

        assert AGENT_REGISTRY["GitHubMCPAgent"].extra["requires_service"] == ["github"]
        assert AGENT_REGISTRY["NotionMCPAgent"].extra["requires_service"] == ["notion"]
