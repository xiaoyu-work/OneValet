"""
MCP Tool Provider - Bridge between MCP servers and OneValet ToolRegistry

Automatically converts MCP tools to OneValet tools and registers them.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from ..tools.registry import ToolRegistry
from ..tools.models import ToolDefinition, ToolCategory, ToolExecutionContext
from .protocol import MCPClientProtocol
from .models import MCPTool

logger = logging.getLogger(__name__)


class MCPToolProvider:
    """
    Bridges MCP servers to OneValet's ToolRegistry

    Converts MCP tools to OneValet ToolDefinitions and registers them.
    Tool names are prefixed with "mcp__{server_name}__" to avoid conflicts.

    Example:
        # Connect to MCP server
        client = MyMCPClient(config)
        await client.connect()

        # Create provider and register tools
        provider = MCPToolProvider(client)
        await provider.register_tools()

        # Tools are now available with prefix: mcp__myserver__toolname
        tools = provider.get_tool_names()

        # Use with ToolExecutor
        executor = ToolExecutor(llm_client=llm)
        result = await executor.run_with_tools(
            messages=[...],
            tool_names=tools,
            context=context
        )

        # Cleanup
        await provider.unregister_tools()
        await client.disconnect()
    """

    def __init__(
        self,
        client: MCPClientProtocol,
        registry: Optional[ToolRegistry] = None,
        tool_prefix: str = "mcp"
    ):
        """
        Initialize MCP tool provider

        Args:
            client: MCP client implementing MCPClientProtocol
            registry: ToolRegistry to register tools (defaults to singleton)
            tool_prefix: Prefix for tool names (default: "mcp")
        """
        self.client = client
        self.registry = registry or ToolRegistry.get_instance()
        self.tool_prefix = tool_prefix
        self._registered_tools: List[str] = []

    def _make_tool_name(self, mcp_tool: MCPTool) -> str:
        """Generate OneValet tool name from MCP tool"""
        return f"{self.tool_prefix}__{self.client.server_name}__{mcp_tool.name}"

    def _create_tool_executor(self, mcp_tool: MCPTool):
        """
        Create an async executor function for the MCP tool

        This wraps the MCP client's call_tool method.
        """
        client = self.client
        tool_name = mcp_tool.name

        async def executor(context: ToolExecutionContext, **kwargs) -> Dict[str, Any]:
            """Execute MCP tool"""
            logger.debug(f"Executing MCP tool: {tool_name} with args: {kwargs}")

            result = await client.call_tool(tool_name, kwargs)

            if result.is_error:
                raise RuntimeError(f"MCP tool error: {result.error_message}")

            # Convert result to dict if needed
            content = result.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {"result": content}
            elif not isinstance(content, dict):
                content = {"result": str(content)}

            return content

        return executor

    async def register_tools(self) -> List[str]:
        """
        Fetch tools from MCP server and register them with ToolRegistry

        Returns:
            List of registered tool names

        Raises:
            ConnectionError: If not connected to MCP server
        """
        if not self.client.is_connected:
            raise ConnectionError(
                f"MCP client not connected. Call client.connect() first."
            )

        # Get tools from MCP server
        mcp_tools = await self.client.list_tools()
        logger.info(f"Found {len(mcp_tools)} tools from MCP server: {self.client.server_name}")

        registered = []
        for mcp_tool in mcp_tools:
            tool_name = self._make_tool_name(mcp_tool)

            # Create ToolDefinition
            tool_def = ToolDefinition(
                name=tool_name,
                description=f"[MCP:{self.client.server_name}] {mcp_tool.description}",
                parameters=mcp_tool.input_schema,
                executor=self._create_tool_executor(mcp_tool),
                category=ToolCategory.CUSTOM,  # Use CUSTOM for MCP tools
            )

            # Register with ToolRegistry
            self.registry.register(tool_def)
            self._registered_tools.append(tool_name)
            registered.append(tool_name)

            logger.debug(f"Registered MCP tool: {tool_name}")

        logger.info(f"Registered {len(registered)} MCP tools from {self.client.server_name}")
        return registered

    async def unregister_tools(self) -> None:
        """Unregister all tools from this MCP provider"""
        for tool_name in self._registered_tools:
            self.registry.unregister(tool_name)
            logger.debug(f"Unregistered MCP tool: {tool_name}")

        logger.info(f"Unregistered {len(self._registered_tools)} MCP tools")
        self._registered_tools = []

    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names from this provider"""
        return list(self._registered_tools)

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI-format tool schemas for registered tools"""
        return self.registry.get_tools_schema(self._registered_tools)

    async def refresh_tools(self) -> List[str]:
        """
        Refresh tools from MCP server

        Unregisters existing tools and re-fetches from server.

        Returns:
            List of newly registered tool names
        """
        await self.unregister_tools()
        return await self.register_tools()

    def __repr__(self) -> str:
        return (
            f"MCPToolProvider(server='{self.client.server_name}', "
            f"tools={len(self._registered_tools)})"
        )


class MCPManager:
    """
    Manages multiple MCP server connections and their tools

    Example:
        manager = MCPManager()

        # Add servers
        await manager.add_server(filesystem_client)
        await manager.add_server(database_client)

        # Get all tool names
        all_tools = manager.get_all_tool_names()

        # Use with ToolExecutor
        result = await executor.run_with_tools(
            messages=[...],
            tool_names=all_tools,
            context=context
        )

        # Cleanup
        await manager.disconnect_all()
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        """
        Initialize MCP manager

        Args:
            registry: ToolRegistry to use (defaults to singleton)
        """
        self.registry = registry or ToolRegistry.get_instance()
        self._providers: Dict[str, MCPToolProvider] = {}

    async def add_server(
        self,
        client: MCPClientProtocol,
        connect: bool = True
    ) -> MCPToolProvider:
        """
        Add an MCP server and register its tools

        Args:
            client: MCP client (already connected or will be connected)
            connect: Whether to connect if not already connected

        Returns:
            MCPToolProvider for the server
        """
        server_name = client.server_name

        if server_name in self._providers:
            logger.warning(f"Server {server_name} already added, replacing")
            await self.remove_server(server_name)

        if connect and not client.is_connected:
            await client.connect()

        provider = MCPToolProvider(client, self.registry)
        await provider.register_tools()

        self._providers[server_name] = provider
        logger.info(f"Added MCP server: {server_name}")

        return provider

    async def remove_server(self, server_name: str) -> None:
        """
        Remove an MCP server and unregister its tools

        Args:
            server_name: Name of the server to remove
        """
        if server_name not in self._providers:
            logger.warning(f"Server {server_name} not found")
            return

        provider = self._providers[server_name]
        await provider.unregister_tools()
        await provider.client.disconnect()

        del self._providers[server_name]
        logger.info(f"Removed MCP server: {server_name}")

    def get_provider(self, server_name: str) -> Optional[MCPToolProvider]:
        """Get provider for a specific server"""
        return self._providers.get(server_name)

    def get_all_tool_names(self) -> List[str]:
        """Get all registered tool names from all servers"""
        names = []
        for provider in self._providers.values():
            names.extend(provider.get_tool_names())
        return names

    def get_server_tool_names(self, server_name: str) -> List[str]:
        """Get tool names from a specific server"""
        provider = self._providers.get(server_name)
        return provider.get_tool_names() if provider else []

    async def refresh_all(self) -> Dict[str, List[str]]:
        """
        Refresh tools from all servers

        Returns:
            Dict mapping server_name to list of tool names
        """
        result = {}
        for server_name, provider in self._providers.items():
            result[server_name] = await provider.refresh_tools()
        return result

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers"""
        for server_name in list(self._providers.keys()):
            await self.remove_server(server_name)

    def __repr__(self) -> str:
        servers = list(self._providers.keys())
        return f"MCPManager(servers={servers})"
