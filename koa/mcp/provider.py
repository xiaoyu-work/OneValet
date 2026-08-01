"""
MCP Tool Provider - Bridge between MCP servers and Koa AgentTool

Automatically converts MCP tools to AgentTool instances.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

from ..models import AgentTool, AgentToolContext
from .models import MCPTool
from .protocol import MCPClientProtocol
from .registry import MCPToolSpec

logger = logging.getLogger(__name__)


class MCPToolProvider:
    """
    Bridges MCP servers to Koa's AgentTool system.

    Converts MCP tools to AgentTool instances.
    Tool names are prefixed with "mcp__{server_name}__" to avoid conflicts.

    Optionally takes ``tool_specs``, a pinned allowlist mapping vendor tool names
    to :class:`~koa.mcp.registry.MCPToolSpec`.  The MCP protocol carries no risk
    classification, so without it every remote tool would arrive as an
    unapproved read and slip past the approval gate; tools missing from the map
    are dropped so vendor drift cannot silently widen the tool surface.

    Example:
        client = MyMCPClient(config)
        await client.connect()

        provider = MCPToolProvider(client)
        tools = await provider.discover_tools()
        # tools is List[AgentTool] — add to orchestrator's builtin_tools
    """

    def __init__(
        self,
        client: MCPClientProtocol,
        tool_prefix: str = "mcp",
        tool_specs: Optional[Dict[str, MCPToolSpec]] = None,
    ):
        self.client = client
        self.tool_prefix = tool_prefix
        self.tool_specs = tool_specs
        self._tools: List[AgentTool] = []

    def _make_tool_name(self, mcp_tool: MCPTool) -> str:
        """Generate Koa tool name from MCP tool"""
        return f"{self.tool_prefix}__{self.client.server_name}__{mcp_tool.name}"

    def _create_tool_executor(self, mcp_tool: MCPTool):
        """Create an async executor function for the MCP tool."""
        client = self.client
        tool_name = mcp_tool.name

        async def executor(args: dict, context: AgentToolContext = None) -> str:
            logger.debug(f"Executing MCP tool: {tool_name} with args: {args}")
            result = await client.call_tool(tool_name, args)
            if result.is_error:
                return f"Error: {result.error_message}"
            content = result.content
            if isinstance(content, dict):
                return json.dumps(content)
            return str(content)

        return executor

    async def discover_tools(self) -> List[AgentTool]:
        """
        Fetch tools from MCP server and return as AgentTool instances.

        When ``tool_specs`` is set, tools absent from it are dropped and the
        remaining ones inherit its risk level and approval requirement.

        Returns:
            List of AgentTool instances
        """
        if not self.client.is_connected:
            raise ConnectionError("MCP client not connected. Call client.connect() first.")

        mcp_tools = await self.client.list_tools()
        logger.info(f"Found {len(mcp_tools)} tools from MCP server: {self.client.server_name}")

        self._tools = []
        skipped: List[str] = []
        for mcp_tool in mcp_tools:
            spec: Optional[MCPToolSpec] = None
            if self.tool_specs is not None:
                spec = self.tool_specs.get(mcp_tool.name)
                if spec is None:
                    skipped.append(mcp_tool.name)
                    continue

            tool = AgentTool(
                name=self._make_tool_name(mcp_tool),
                description=f"[MCP:{self.client.server_name}] {mcp_tool.description}",
                parameters=mcp_tool.input_schema,
                executor=self._create_tool_executor(mcp_tool),
                category="mcp",
                risk_level=spec.risk_level if spec else "read",
                needs_approval=spec.needs_approval if spec else False,
                sensitive_args=list(spec.sensitive_args) if spec else [],
            )
            self._tools.append(tool)
            logger.debug(f"Created MCP tool: {tool.name}")

        if skipped:
            logger.info(
                "Dropped %d unpinned tool(s) from %s: %s",
                len(skipped),
                self.client.server_name,
                ", ".join(sorted(skipped)),
            )
        missing = self._missing_pinned_tools({t.name for t in mcp_tools})
        if missing:
            # The vendor removed or renamed a tool we depend on — surface it
            # loudly, since the agent silently loses that capability.
            logger.warning(
                "Pinned tool(s) missing from %s: %s",
                self.client.server_name,
                ", ".join(sorted(missing)),
            )

        logger.info(f"Created {len(self._tools)} MCP tools from {self.client.server_name}")
        return list(self._tools)

    def _missing_pinned_tools(self, advertised: set) -> Tuple[str, ...]:
        """Pinned tool names the server did not advertise."""
        if not self.tool_specs:
            return ()
        return tuple(name for name in self.tool_specs if name not in advertised)

    def get_tools(self) -> List[AgentTool]:
        """Get previously discovered tools."""
        return list(self._tools)

    def get_tool_names(self) -> List[str]:
        """Get list of tool names from this provider."""
        return [t.name for t in self._tools]

    async def refresh_tools(self) -> List[AgentTool]:
        """Re-fetch tools from MCP server."""
        return await self.discover_tools()

    def __repr__(self) -> str:
        return f"MCPToolProvider(server='{self.client.server_name}', tools={len(self._tools)})"


class MCPManager:
    """
    Manages multiple MCP server connections and their tools.

    Connections are keyed by ``(tenant_id, server_name)``.  Keying by server name
    alone would hand tenant B the session tenant A already authenticated, so a
    tenant id is required whenever the servers carry per-user credentials.

    Example:
        manager = MCPManager()
        await manager.add_server(filesystem_client, tenant_id="t1")
        tools = manager.get_all_tools(tenant_id="t1")  # List[AgentTool]
    """

    def __init__(self):
        self._providers: Dict[Tuple[str, str], MCPToolProvider] = {}

    @staticmethod
    def _key(tenant_id: str, server_name: str) -> Tuple[str, str]:
        return (tenant_id or "", server_name)

    async def add_server(
        self,
        client: MCPClientProtocol,
        connect: bool = True,
        tenant_id: str = "",
        tool_specs: Optional[Dict[str, MCPToolSpec]] = None,
    ) -> MCPToolProvider:
        """Add an MCP server for a tenant and discover its tools."""
        server_name = client.server_name
        key = self._key(tenant_id, server_name)

        if key in self._providers:
            logger.warning(f"Server {server_name} already added for tenant, replacing")
            await self.remove_server(server_name, tenant_id=tenant_id)

        if connect and not client.is_connected:
            await client.connect()

        provider = MCPToolProvider(client, tool_specs=tool_specs)
        await provider.discover_tools()

        self._providers[key] = provider
        logger.info(f"Added MCP server: {server_name}")
        return provider

    async def remove_server(self, server_name: str, tenant_id: str = "") -> None:
        """Remove an MCP server for a tenant."""
        key = self._key(tenant_id, server_name)
        if key not in self._providers:
            logger.warning(f"Server {server_name} not found")
            return

        provider = self._providers[key]
        await provider.client.disconnect()
        del self._providers[key]
        logger.info(f"Removed MCP server: {server_name}")

    def get_provider(
        self, server_name: str, tenant_id: str = ""
    ) -> Optional[MCPToolProvider]:
        """Get provider for a specific server within a tenant."""
        return self._providers.get(self._key(tenant_id, server_name))

    def _providers_for(self, tenant_id: Optional[str]):
        """Providers for one tenant, or all of them when tenant_id is None."""
        if tenant_id is None:
            return list(self._providers.values())
        return [p for (tid, _), p in self._providers.items() if tid == (tenant_id or "")]

    def get_all_tools(self, tenant_id: Optional[str] = None) -> List[AgentTool]:
        """Get AgentTool instances, scoped to a tenant unless None is passed."""
        tools = []
        for provider in self._providers_for(tenant_id):
            tools.extend(provider.get_tools())
        return tools

    def get_all_tool_names(self, tenant_id: Optional[str] = None) -> List[str]:
        """Get all tool names, scoped to a tenant unless None is passed."""
        return [t.name for t in self.get_all_tools(tenant_id)]

    async def refresh_all(
        self, tenant_id: Optional[str] = None
    ) -> Dict[str, List[AgentTool]]:
        """Refresh tools from all servers, scoped to a tenant unless None."""
        result = {}
        for (tid, server_name), provider in list(self._providers.items()):
            if tenant_id is not None and tid != (tenant_id or ""):
                continue
            result[server_name] = await provider.refresh_tools()
        return result

    async def disconnect_all(self, tenant_id: Optional[str] = None) -> None:
        """Disconnect from MCP servers, scoped to a tenant unless None."""
        for tid, server_name in list(self._providers.keys()):
            if tenant_id is not None and tid != (tenant_id or ""):
                continue
            await self.remove_server(server_name, tenant_id=tid)

    def __repr__(self) -> str:
        servers = [f"{tid}:{name}" for tid, name in self._providers]
        return f"MCPManager(servers={servers})"
