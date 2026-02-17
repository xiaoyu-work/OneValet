"""
MCP Tool Provider - Bridge between MCP servers and OneValet AgentTool

Automatically converts MCP tools to AgentTool instances.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from ..models import AgentTool, AgentToolContext
from .protocol import MCPClientProtocol
from .models import MCPTool

logger = logging.getLogger(__name__)


class MCPToolProvider:
    """
    Bridges MCP servers to OneValet's AgentTool system.

    Converts MCP tools to AgentTool instances.
    Tool names are prefixed with "mcp__{server_name}__" to avoid conflicts.

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
        tool_prefix: str = "mcp"
    ):
        self.client = client
        self.tool_prefix = tool_prefix
        self._tools: List[AgentTool] = []

    def _make_tool_name(self, mcp_tool: MCPTool) -> str:
        """Generate OneValet tool name from MCP tool"""
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

        Returns:
            List of AgentTool instances
        """
        if not self.client.is_connected:
            raise ConnectionError(
                f"MCP client not connected. Call client.connect() first."
            )

        mcp_tools = await self.client.list_tools()
        logger.info(f"Found {len(mcp_tools)} tools from MCP server: {self.client.server_name}")

        self._tools = []
        for mcp_tool in mcp_tools:
            tool = AgentTool(
                name=self._make_tool_name(mcp_tool),
                description=f"[MCP:{self.client.server_name}] {mcp_tool.description}",
                parameters=mcp_tool.input_schema,
                executor=self._create_tool_executor(mcp_tool),
                category="mcp",
            )
            self._tools.append(tool)
            logger.debug(f"Created MCP tool: {tool.name}")

        logger.info(f"Created {len(self._tools)} MCP tools from {self.client.server_name}")
        return list(self._tools)

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
        return (
            f"MCPToolProvider(server='{self.client.server_name}', "
            f"tools={len(self._tools)})"
        )


class MCPManager:
    """
    Manages multiple MCP server connections and their tools.

    Example:
        manager = MCPManager()
        await manager.add_server(filesystem_client)
        tools = manager.get_all_tools()  # List[AgentTool]
    """

    def __init__(self):
        self._providers: Dict[str, MCPToolProvider] = {}

    async def add_server(
        self,
        client: MCPClientProtocol,
        connect: bool = True
    ) -> MCPToolProvider:
        """Add an MCP server and discover its tools."""
        server_name = client.server_name

        if server_name in self._providers:
            logger.warning(f"Server {server_name} already added, replacing")
            await self.remove_server(server_name)

        if connect and not client.is_connected:
            await client.connect()

        provider = MCPToolProvider(client)
        await provider.discover_tools()

        self._providers[server_name] = provider
        logger.info(f"Added MCP server: {server_name}")
        return provider

    async def remove_server(self, server_name: str) -> None:
        """Remove an MCP server."""
        if server_name not in self._providers:
            logger.warning(f"Server {server_name} not found")
            return

        provider = self._providers[server_name]
        await provider.client.disconnect()
        del self._providers[server_name]
        logger.info(f"Removed MCP server: {server_name}")

    def get_provider(self, server_name: str) -> Optional[MCPToolProvider]:
        """Get provider for a specific server."""
        return self._providers.get(server_name)

    def get_all_tools(self) -> List[AgentTool]:
        """Get all AgentTool instances from all servers."""
        tools = []
        for provider in self._providers.values():
            tools.extend(provider.get_tools())
        return tools

    def get_all_tool_names(self) -> List[str]:
        """Get all tool names from all servers."""
        return [t.name for t in self.get_all_tools()]

    async def refresh_all(self) -> Dict[str, List[AgentTool]]:
        """Refresh tools from all servers."""
        result = {}
        for server_name, provider in self._providers.items():
            result[server_name] = await provider.refresh_tools()
        return result

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for server_name in list(self._providers.keys()):
            await self.remove_server(server_name)

    def __repr__(self) -> str:
        servers = list(self._providers.keys())
        return f"MCPManager(servers={servers})"
