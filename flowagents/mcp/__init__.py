"""
FlowAgent MCP Integration - Model Context Protocol support

Allows MCP servers to be registered as tool providers for agents.
MCP tools are automatically converted to FlowAgent tools and registered
in the ToolRegistry.

Example:
    from flowagents.mcp import MCPClient, MCPToolProvider

    # Connect to MCP server
    client = MCPClient(transport="stdio", command=["python", "my_mcp_server.py"])
    await client.connect()

    # Register MCP tools with an agent
    provider = MCPToolProvider(client)
    await provider.register_tools()

    # Now MCP tools are available in ToolRegistry
    executor = ToolExecutor(llm_client=my_llm)
    result = await executor.run_with_tools(
        messages=[...],
        tool_names=["mcp__my_tool"],  # MCP tools are prefixed with "mcp__"
        context=context
    )
"""

from .protocol import MCPClientProtocol, MCPTransport
from .client import MCPClient, MockMCPClient
from .provider import MCPToolProvider, MCPManager
from .models import MCPServerConfig, MCPTool, MCPResource, MCPCallResult

__all__ = [
    "MCPClientProtocol",
    "MCPTransport",
    "MCPClient",
    "MockMCPClient",
    "MCPToolProvider",
    "MCPManager",
    "MCPServerConfig",
    "MCPTool",
    "MCPResource",
    "MCPCallResult",
]
