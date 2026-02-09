"""
MCP Protocol - Abstract interface for MCP clients

Users can implement this protocol to integrate any MCP client library.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from enum import Enum

from .models import MCPTool, MCPResource, MCPCallResult, MCPPrompt


class MCPTransport(str, Enum):
    """MCP transport types"""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


@runtime_checkable
class MCPClientProtocol(Protocol):
    """
    Abstract interface for MCP clients

    Implement this protocol to integrate with any MCP client library
    (e.g., official MCP SDK, custom implementation).

    Example:
        class MyMCPClient:
            async def connect(self) -> None:
                # Connect to MCP server
                pass

            async def disconnect(self) -> None:
                # Disconnect from MCP server
                pass

            async def list_tools(self) -> List[MCPTool]:
                # List available tools
                pass

            async def call_tool(self, name: str, arguments: Dict) -> MCPCallResult:
                # Call a tool
                pass

            # ... other methods
    """

    @property
    def server_name(self) -> str:
        """Get the server name"""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to server"""
        ...

    async def connect(self) -> None:
        """
        Connect to the MCP server

        Raises:
            ConnectionError: If connection fails
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        ...

    async def list_tools(self) -> List[MCPTool]:
        """
        List all available tools from the MCP server

        Returns:
            List of MCPTool objects
        """
        ...

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
            MCPCallResult with the tool output
        """
        ...

    async def list_resources(self) -> List[MCPResource]:
        """
        List all available resources from the MCP server

        Returns:
            List of MCPResource objects
        """
        ...

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource from the MCP server

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        ...

    async def list_prompts(self) -> List[MCPPrompt]:
        """
        List all available prompts from the MCP server

        Returns:
            List of MCPPrompt objects
        """
        ...

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get a rendered prompt from the MCP server

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            Rendered prompt string
        """
        ...
