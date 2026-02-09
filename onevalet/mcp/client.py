"""
MCP Client - Base implementation for MCP server communication

This provides a reference implementation. Users can:
1. Use this directly with the official MCP SDK
2. Implement MCPClientProtocol for custom integrations
"""

import logging
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from .models import (
    MCPServerConfig,
    MCPTool,
    MCPResource,
    MCPCallResult,
    MCPPrompt,
    MCPTransportType,
)
from .protocol import MCPClientProtocol

logger = logging.getLogger(__name__)


class MCPClient(MCPClientProtocol):
    """
    MCP Client implementation

    This is a base implementation that can be extended or replaced.
    For production use, integrate with the official MCP SDK.

    Example:
        config = MCPServerConfig(
            name="my-server",
            transport=MCPTransportType.STDIO,
            command="python",
            args=["my_mcp_server.py"]
        )
        client = MCPClient(config)
        await client.connect()

        tools = await client.list_tools()
        result = await client.call_tool("my_tool", {"arg": "value"})

        await client.disconnect()
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client

        Args:
            config: MCP server configuration
        """
        self.config = config
        self._connected = False
        self._tools: List[MCPTool] = []
        self._resources: List[MCPResource] = []
        self._prompts: List[MCPPrompt] = []

        # Transport-specific client (to be set by subclass or connect())
        self._transport = None

    @property
    def server_name(self) -> str:
        return self.config.name

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """
        Connect to the MCP server

        Override this method or use a subclass for specific transport implementations.
        """
        if self._connected:
            logger.warning(f"Already connected to {self.server_name}")
            return

        logger.info(f"Connecting to MCP server: {self.server_name}")

        try:
            if self.config.transport == MCPTransportType.STDIO:
                await self._connect_stdio()
            elif self.config.transport == MCPTransportType.SSE:
                await self._connect_sse()
            elif self.config.transport == MCPTransportType.WEBSOCKET:
                await self._connect_websocket()

            self._connected = True
            logger.info(f"Connected to MCP server: {self.server_name}")

            # Discover available tools, resources, prompts
            await self._discover_capabilities()

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.server_name}: {e}")
            raise ConnectionError(f"MCP connection failed: {e}") from e

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport - override for actual implementation"""
        # This is a placeholder - integrate with MCP SDK for real implementation
        raise NotImplementedError(
            "STDIO transport requires MCP SDK integration. "
            "Override _connect_stdio() or use MCPSDKClient."
        )

    async def _connect_sse(self) -> None:
        """Connect via SSE transport - override for actual implementation"""
        raise NotImplementedError(
            "SSE transport requires MCP SDK integration. "
            "Override _connect_sse() or use MCPSDKClient."
        )

    async def _connect_websocket(self) -> None:
        """Connect via WebSocket transport - override for actual implementation"""
        raise NotImplementedError(
            "WebSocket transport requires MCP SDK integration. "
            "Override _connect_websocket() or use MCPSDKClient."
        )

    async def _discover_capabilities(self) -> None:
        """Discover tools, resources, and prompts from server"""
        try:
            self._tools = await self._fetch_tools()
            logger.info(f"Discovered {len(self._tools)} tools from {self.server_name}")
        except Exception as e:
            logger.warning(f"Failed to discover tools: {e}")

        try:
            self._resources = await self._fetch_resources()
            logger.info(f"Discovered {len(self._resources)} resources from {self.server_name}")
        except Exception as e:
            logger.warning(f"Failed to discover resources: {e}")

        try:
            self._prompts = await self._fetch_prompts()
            logger.info(f"Discovered {len(self._prompts)} prompts from {self.server_name}")
        except Exception as e:
            logger.warning(f"Failed to discover prompts: {e}")

    async def _fetch_tools(self) -> List[MCPTool]:
        """Fetch tools from server - override for actual implementation"""
        return []

    async def _fetch_resources(self) -> List[MCPResource]:
        """Fetch resources from server - override for actual implementation"""
        return []

    async def _fetch_prompts(self) -> List[MCPPrompt]:
        """Fetch prompts from server - override for actual implementation"""
        return []

    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        if not self._connected:
            return

        logger.info(f"Disconnecting from MCP server: {self.server_name}")
        self._connected = False
        self._tools = []
        self._resources = []
        self._prompts = []

    async def list_tools(self) -> List[MCPTool]:
        """List all available tools"""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> MCPCallResult:
        """
        Call a tool on the MCP server

        Args:
            name: Tool name (without server prefix)
            arguments: Tool arguments

        Returns:
            MCPCallResult
        """
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")

        # Find the tool
        tool = next((t for t in self._tools if t.name == name), None)
        if not tool:
            return MCPCallResult(
                content=None,
                is_error=True,
                error_message=f"Unknown tool: {name}"
            )

        try:
            result = await self._execute_tool(name, arguments)
            return MCPCallResult(content=result)
        except Exception as e:
            logger.error(f"Tool execution failed: {name} - {e}")
            return MCPCallResult(
                content=None,
                is_error=True,
                error_message=str(e)
            )

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute tool - override for actual implementation"""
        raise NotImplementedError("Override _execute_tool() for actual implementation")

    async def list_resources(self) -> List[MCPResource]:
        """List all available resources"""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        return self._resources

    async def read_resource(self, uri: str) -> Any:
        """Read a resource"""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        raise NotImplementedError("Override read_resource() for actual implementation")

    async def list_prompts(self) -> List[MCPPrompt]:
        """List all available prompts"""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        return self._prompts

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get a rendered prompt"""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        raise NotImplementedError("Override get_prompt() for actual implementation")

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"MCPClient(server='{self.server_name}', status={status})"


class MockMCPClient(MCPClient):
    """
    Mock MCP client for testing

    Example:
        client = MockMCPClient(
            name="test-server",
            tools=[
                MCPTool(
                    name="echo",
                    description="Echo input",
                    input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                    server_name="test-server"
                )
            ]
        )
        await client.connect()
    """

    def __init__(
        self,
        name: str = "mock-server",
        tools: Optional[List[MCPTool]] = None,
        resources: Optional[List[MCPResource]] = None,
        tool_handler: Optional[callable] = None
    ):
        config = MCPServerConfig(
            name=name,
            transport=MCPTransportType.STDIO,
            command="mock"
        )
        super().__init__(config)
        self._mock_tools = tools or []
        self._mock_resources = resources or []
        self._tool_handler = tool_handler

    async def _connect_stdio(self) -> None:
        """Mock connection"""
        pass

    async def _fetch_tools(self) -> List[MCPTool]:
        return self._mock_tools

    async def _fetch_resources(self) -> List[MCPResource]:
        return self._mock_resources

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if self._tool_handler:
            return await self._tool_handler(name, arguments)
        return {"result": f"Mock result for {name}", "arguments": arguments}
