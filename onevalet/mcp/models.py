"""
MCP Models - Data structures for MCP integration
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class MCPTransportType(str, Enum):
    """MCP transport types"""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


@dataclass
class MCPServerConfig:
    """
    Configuration for connecting to an MCP server

    Attributes:
        name: Unique name for this MCP server
        transport: Transport type (stdio, sse, websocket)
        command: Command to start server (for stdio)
        args: Command arguments
        url: Server URL (for sse/websocket)
        env: Environment variables
        headers: HTTP headers (for sse/websocket)

    Example (stdio):
        config = MCPServerConfig(
            name="filesystem",
            transport=MCPTransportType.STDIO,
            command="npx",
            args=["-y", "@anthropic/mcp-server-filesystem", "/path/to/dir"]
        )

    Example (SSE):
        config = MCPServerConfig(
            name="remote-api",
            transport=MCPTransportType.SSE,
            url="http://localhost:8080/mcp"
        )
    """
    name: str
    transport: MCPTransportType = MCPTransportType.STDIO
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0

    def __post_init__(self):
        if self.transport == MCPTransportType.STDIO and not self.command:
            raise ValueError("STDIO transport requires 'command'")
        if self.transport in (MCPTransportType.SSE, MCPTransportType.WEBSOCKET) and not self.url:
            raise ValueError(f"{self.transport.value} transport requires 'url'")


@dataclass
class MCPTool:
    """
    Represents a tool from an MCP server

    Attributes:
        name: Tool name (as defined by MCP server)
        description: Tool description
        input_schema: JSON Schema for tool parameters
        server_name: Name of the MCP server providing this tool
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str

    @property
    def full_name(self) -> str:
        """Full tool name with server prefix: mcp__{server}__{tool}"""
        return f"mcp__{self.server_name}__{self.name}"

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.full_name,
                "description": f"[MCP:{self.server_name}] {self.description}",
                "parameters": self.input_schema
            }
        }


@dataclass
class MCPResource:
    """
    Represents a resource from an MCP server

    Attributes:
        uri: Resource URI
        name: Human-readable name
        description: Resource description
        mime_type: MIME type of the resource
        server_name: Name of the MCP server providing this resource
    """
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    server_name: str = ""


@dataclass
class MCPCallResult:
    """Result from calling an MCP tool"""
    content: Any
    is_error: bool = False
    error_message: Optional[str] = None


@dataclass
class MCPPrompt:
    """
    Represents a prompt template from an MCP server

    Attributes:
        name: Prompt name
        description: Prompt description
        arguments: List of argument definitions
        server_name: Name of the MCP server
    """
    name: str
    description: Optional[str] = None
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    server_name: str = ""
